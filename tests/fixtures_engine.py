"""Engine test fixtures: self-contained definitions (no dependence on emberdeep.data content) and world builders."""
from __future__ import annotations

import numpy as np

from emberdeep.core import factory as F
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.types import (
    Ability as A, ActionCost as AC, AreaShape, ArmorCategory, AttackKind, Condition, ControllerKind, CreatureType,
    DamageType as DT, DoorState, Material, Relation, School, Size, Skill, Terrain, TargetType, WeaponCategory,
    WeaponProperty as WP,
)
from emberdeep.core.world import World, RulesConfig
from emberdeep.data.schema import (
    AreaSpec, AttackDef, CharacterBuild, ClassDef, ConditionSpec, DamageSpec, EffectSpec, FeatureDef, ItemDef,
    MonsterDef, RaceDef, ResourceDef, SaveActionDef, SaveSpec, SpecialActionDef, SpellDef, SpellcastingDef,
    SpellcastingProgression, SubclassDef, WeaponDef, ArmorDef, ZoneSpec,
)
from emberdeep.scenarios.mapdata import DoorData, MapData, ObjectData

# ------------------------------------------------------------------------------------------ equipment
WEAPONS = [
    WeaponDef("longsword", "Longsword", WeaponCategory.MARTIAL_MELEE, "1d8", DT.SLASHING, WP.VERSATILE, versatile="1d10"),
    WeaponDef("greatsword", "Greatsword", WeaponCategory.MARTIAL_MELEE, "2d6", DT.SLASHING, WP.HEAVY | WP.TWO_HANDED),
    WeaponDef("dagger", "Dagger", WeaponCategory.SIMPLE_MELEE, "1d4", DT.PIERCING,
              WP.FINESSE | WP.LIGHT | WP.THROWN, range_ft=(20, 60)),
    WeaponDef("shortsword", "Shortsword", WeaponCategory.MARTIAL_MELEE, "1d6", DT.PIERCING, WP.FINESSE | WP.LIGHT),
    WeaponDef("handaxe", "Handaxe", WeaponCategory.SIMPLE_MELEE, "1d6", DT.SLASHING, WP.LIGHT | WP.THROWN,
              range_ft=(20, 60)),
    WeaponDef("mace", "Mace", WeaponCategory.SIMPLE_MELEE, "1d6", DT.BLUDGEONING),
    WeaponDef("quarterstaff", "Quarterstaff", WeaponCategory.SIMPLE_MELEE, "1d6", DT.BLUDGEONING, WP.VERSATILE,
              versatile="1d8"),
    WeaponDef("shortbow", "Shortbow", WeaponCategory.SIMPLE_RANGED, "1d6", DT.PIERCING,
              WP.AMMUNITION | WP.RANGED | WP.TWO_HANDED, range_ft=(80, 320)),
    WeaponDef("glaive", "Glaive", WeaponCategory.MARTIAL_MELEE, "1d10", DT.SLASHING,
              WP.HEAVY | WP.REACH | WP.TWO_HANDED, reach_ft=10),
    WeaponDef("light_crossbow", "Light Crossbow", WeaponCategory.SIMPLE_RANGED, "1d8", DT.PIERCING,
              WP.AMMUNITION | WP.LOADING | WP.RANGED | WP.TWO_HANDED, range_ft=(80, 320)),
]
ARMORS = [
    ArmorDef("leather", "Leather", ArmorCategory.LIGHT, 11, None),
    ArmorDef("chain_shirt", "Chain Shirt", ArmorCategory.MEDIUM, 13, 2),
    ArmorDef("chain_mail", "Chain Mail", ArmorCategory.HEAVY, 16, 0, str_req=13, stealth_disadvantage=True),
    ArmorDef("shield", "Shield", ArmorCategory.SHIELD, 2),
]
ITEMS = [
    ItemDef("potion_of_healing", "Potion of Healing", "potion", 50, consumable=True, heal="2d4+2"),
    ItemDef("thieves_tools", "Thieves' Tools", "tool", 25),
    ItemDef("torch", "Torch", "light", 0.01, light_radius_ft=(20, 20)),
    ItemDef("arrows", "Arrows", "ammo", 1),
    ItemDef("key_a", "Brass Key", "key", 1),
    ItemDef("gem", "Gem", "gem", 100),
    ItemDef("scroll_fire_bolt", "Scroll of Fire Bolt", "scroll", 25, consumable=True, use_spell="fire_bolt"),
    ItemDef("wand_mm", "Wand of Magic Missiles", "wand", 500, charges=3, use_spell="magic_missile", use_spell_level=1),
    ItemDef("alchemists_fire", "Alchemist's Fire", "consumable", 50, consumable=True,
            damage=DamageSpec("1d4", DT.FIRE), range_ft=(20, 20)),
    ItemDef("longsword_p1", "Longsword +1", "weapon", 500, base="longsword", bonus=1),
]

# ------------------------------------------------------------------------------------------ spells
def _d(dice, dt, per="", cant=False, mod=False):
    return DamageSpec(dice, dt, add_mod=mod, per_slot_above=per, cantrip_scaling=cant, magical=True)


SPELLS = [
    SpellDef("fire_bolt", "Fire Bolt", 0, School.EVOCATION, ("wizard",), range_ft=120, attack=AttackKind.RANGED_SPELL,
             damage=(_d("1d10", DT.FIRE, cant=True),), vfx="fire_bolt"),
    SpellDef("magic_missile", "Magic Missile", 1, School.EVOCATION, ("wizard",), range_ft=120, auto_hit=True,
             max_targets=3, targets_per_slot_above=1, damage=(_d("1d4+1", DT.FORCE),), vfx="magic_missile"),
    SpellDef("cure_wounds", "Cure Wounds", 1, School.EVOCATION, ("cleric",), range_ft=5,
             target=TargetType.ANY_CREATURE, healing="1d8", heal_per_slot_above="1d8"),
    SpellDef("healing_word", "Healing Word", 1, School.EVOCATION, ("cleric",), casting_time=AC.BONUS_ACTION,
             range_ft=60, components="V", target=TargetType.ANY_CREATURE, healing="1d4", heal_per_slot_above="1d4"),
    SpellDef("bless", "Bless", 1, School.ENCHANTMENT, ("cleric",), range_ft=30, duration=10, concentration=True,
             target=TargetType.ALLY, max_targets=3, targets_per_slot_above=1, effect=EffectSpec("bless", 10, {"dice": "1d4"})),
    SpellDef("fireball", "Fireball", 3, School.EVOCATION, ("wizard",), range_ft=150, target=TargetType.POINT,
             area=AreaSpec(AreaShape.SPHERE, 20), save=SaveSpec(A.DEX, None, True), damage=(_d("8d6", DT.FIRE, "1d6"),)),
    SpellDef("burning_hands", "Burning Hands", 1, School.EVOCATION, ("wizard",), target=TargetType.NONE,
             area=AreaSpec(AreaShape.CONE, 15, origin="self"), save=SaveSpec(A.DEX, None, True),
             damage=(_d("3d6", DT.FIRE, "1d6"),)),
    SpellDef("web", "Web", 2, School.CONJURATION, ("wizard",), range_ft=60, duration=600, concentration=True,
             target=TargetType.POINT, area=AreaSpec(AreaShape.CUBE, 20), save=SaveSpec(A.DEX),
             conditions=(ConditionSpec(Condition.RESTRAINED, -1, escape_dc=None),),
             zone=ZoneSpec("web", AreaSpec(AreaShape.CUBE, 20), 600, True, {"save": "DEX", "difficult_terrain": True})),
    SpellDef("shield", "Shield", 1, School.ABJURATION, ("wizard",), casting_time=AC.REACTION, duration=1,
             target=TargetType.SELF, hook="shield",
             self_effect=EffectSpec("shield", 1, {"ac": 5, "until": "start_of_turn", "blocks_magic_missile": True})),
    SpellDef("hold_person", "Hold Person", 2, School.ENCHANTMENT, ("cleric",), range_ft=60, duration=10,
             concentration=True, target=TargetType.ENEMY, save=SaveSpec(A.WIS),
             conditions=(ConditionSpec(Condition.PARALYZED, 10, A.WIS),), effect=EffectSpec("hold_person", 10)),
    SpellDef("spirit_guardians", "Spirit Guardians", 3, School.CONJURATION, ("cleric",), duration=100,
             concentration=True, target=TargetType.NONE, area=AreaSpec(AreaShape.EMANATION, 15, origin="self"),
             save=SaveSpec(A.WIS, None, True), damage=(_d("3d8", DT.RADIANT, "1d8"),),
             zone=ZoneSpec("spirit_guardians", AreaSpec(AreaShape.EMANATION, 15, origin="self"), 100, True,
                           {"dice": "3d8", "per_slot": "1d8", "save": "WIS", "half": True, "type": "radiant",
                            "half_speed": True})),
    SpellDef("misty_step", "Misty Step", 2, School.CONJURATION, ("wizard",), casting_time=AC.BONUS_ACTION,
             components="V", target=TargetType.POINT, hook="misty_step"),
    SpellDef("counterspell", "Counterspell", 3, School.ABJURATION, ("wizard",), casting_time=AC.REACTION,
             range_ft=60, components="S", hook="counterspell"),
    SpellDef("hellish_rebuke", "Hellish Rebuke", 1, School.EVOCATION, ("warlock",), casting_time=AC.REACTION,
             range_ft=60, target=TargetType.ENEMY, save=SaveSpec(A.DEX, None, True),
             damage=(_d("2d10", DT.FIRE, "1d10"),), hook="hellish_rebuke"),
    SpellDef("fog_cloud", "Fog Cloud", 1, School.CONJURATION, ("wizard",), range_ft=120, duration=600,
             concentration=True, target=TargetType.POINT, area=AreaSpec(AreaShape.SPHERE, 20),
             zone=ZoneSpec("fog_cloud", AreaSpec(AreaShape.SPHERE, 20), 600, False, {"obscurement": "heavy"})),
    SpellDef("revivify", "Revivify", 3, School.NECROMANCY, ("cleric",), range_ft=5, components="VSM",
             material_cost_gp=300, target=TargetType.CREATURE, hook="revivify"),
]


# ------------------------------------------------------------------------------------------ monsters
def _atk(id_, name, kind, to_hit, dmg, dtype, reach=5, rng=(0, 0), **kw):
    return AttackDef(id_, name, kind, to_hit, reach, rng, (DamageSpec(dmg, dtype),), **kw)


GOBLIN = MonsterDef(
    "goblin", "Goblin", Size.SMALL, CreatureType.HUMANOID, 0.25, 15, "2d6", 7, (8, 14, 10, 10, 8, 8),
    skills={Skill.STEALTH: 6}, darkvision_ft=60, passive_perception=9, traits=("nimble_escape",),
    actions=(_atk("scimitar", "Scimitar", AttackKind.MELEE_WEAPON, 4, "1d6+2", DT.SLASHING),
             _atk("shortbow", "Shortbow", AttackKind.RANGED_WEAPON, 4, "1d6+2", DT.PIERCING, rng=(80, 320))),
    treasure_gp="1d6", morale=0.4, can_surrender=True)
ORC = MonsterDef(
    "orc", "Orc", Size.MEDIUM, CreatureType.HUMANOID, 0.5, 13, "2d8+6", 15, (16, 12, 16, 7, 11, 10), darkvision_ft=60,
    passive_perception=10, actions=(_atk("greataxe", "Greataxe", AttackKind.MELEE_WEAPON, 5, "1d12+3", DT.SLASHING),),
    ai_profile="brute")
WOLF = MonsterDef(
    "wolf", "Wolf", Size.MEDIUM, CreatureType.BEAST, 0.25, 13, "2d8+2", 11, (12, 15, 12, 3, 12, 6),
    speed={"walk": 40}, passive_perception=13, traits=("keen_senses", "pack_tactics"),
    actions=(AttackDef("bite", "Bite", AttackKind.MELEE_WEAPON, 4, 5, (0, 0), (DamageSpec("2d4+2", DT.PIERCING),),
                       on_hit_save=SaveSpec(A.STR, 11), on_hit_conditions=(ConditionSpec(Condition.PRONE, -1),)),))
SKELETON = MonsterDef(
    "skeleton", "Skeleton", Size.MEDIUM, CreatureType.UNDEAD, 0.25, 13, "2d8+4", 13, (10, 14, 15, 6, 8, 5),
    vulnerabilities=(DT.BLUDGEONING,), immunities=(DT.POISON,), condition_immunities=(Condition.POISONED,),
    darkvision_ft=60, passive_perception=9,
    actions=(_atk("shortsword", "Shortsword", AttackKind.MELEE_WEAPON, 4, "1d6+2", DT.PIERCING),))
OGRE = MonsterDef(
    "ogre", "Ogre", Size.LARGE, CreatureType.GIANT, 2, 11, "7d10+21", 59, (19, 8, 16, 5, 7, 7), darkvision_ft=60,
    passive_perception=8, actions=(_atk("greatclub", "Greatclub", AttackKind.MELEE_WEAPON, 6, "2d8+4", DT.BLUDGEONING),))
GIANT_SPIDER = MonsterDef(
    "giant_spider", "Giant Spider", Size.LARGE, CreatureType.BEAST, 1, 14, "4d10+4", 26, (14, 16, 12, 2, 11, 4),
    speed={"walk": 30, "climb": 30}, blindsight_ft=10, darkvision_ft=60, passive_perception=10,
    actions=(AttackDef("bite", "Bite", AttackKind.MELEE_WEAPON, 5, 5, (0, 0), (DamageSpec("1d8+3", DT.PIERCING),),
                       on_hit_save=SaveSpec(A.CON, 11, True), on_hit_save_damage=(DamageSpec("2d8", DT.POISON),)),
             AttackDef("web", "Web", AttackKind.RANGED_WEAPON, 5, 5, (30, 60), (),
                       on_hit_conditions=(ConditionSpec(Condition.RESTRAINED, -1, escape_dc=12),), recharge=5)))
OGRE_MULTI = MonsterDef(
    "bugbear_boss", "Boss", Size.MEDIUM, CreatureType.HUMANOID, 3, 16, "6d8+12", 39, (17, 14, 14, 11, 12, 11),
    actions=(_atk("morningstar", "Morningstar", AttackKind.MELEE_WEAPON, 4, "2d8+2", DT.PIERCING),
             _atk("javelin", "Javelin", AttackKind.RANGED_WEAPON, 4, "1d6+2", DT.PIERCING, rng=(30, 120))),
    multiattack=(("morningstar", "morningstar"),))
DRAGON = MonsterDef(
    "wyrmling", "Wyrmling", Size.MEDIUM, CreatureType.DRAGON, 2, 17, "10d8+10", 55, (19, 10, 13, 14, 11, 15),
    speed={"walk": 30, "fly": 60}, actions=(
        _atk("bite", "Bite", AttackKind.MELEE_WEAPON, 6, "1d10+4", DT.PIERCING),
        SaveActionDef("fire_breath", "Fire Breath", SaveSpec(A.DEX, 13, True), AreaSpec(AreaShape.CONE, 15, origin="self"),
                      damage=(DamageSpec("8d6", DT.FIRE),), recharge=5, target=TargetType.NONE)),
    legendary_per_round=1, legendary_actions=((_atk("tail", "Tail", AttackKind.MELEE_WEAPON, 6, "1d8+4",
                                                     DT.BLUDGEONING), 1),))
MONSTERS = [GOBLIN, ORC, WOLF, SKELETON, OGRE, GIANT_SPIDER, OGRE_MULTI, DRAGON]

# ------------------------------------------------------------------------------------------ classes
def _slots(rows):
    out = [(0,) * 9]
    for r in rows:
        out.append(tuple(list(r) + [0] * (9 - len(r))))
    return tuple(out)


FULL_SLOTS = _slots([(2,), (3,), (4, 2), (4, 3), (4, 3, 2)])
FEATS = [
    FeatureDef("second_wind", "Second Wind", "fighter", 1, "bonus_action", params={"dice": "1d10"}),
    FeatureDef("extra_attack", "Extra Attack", "fighter", 5, params={"attacks": 2}),
    FeatureDef("sneak_attack", "Sneak Attack", "rogue", 1),
    FeatureDef("uncanny_dodge", "Uncanny Dodge", "rogue", 5),
    FeatureDef("divine_smite", "Divine Smite", "paladin", 2),
    FeatureDef("relentless", "Relentless", "race", 1, hook="relentless_endurance"),
]
FIGHTER = ClassDef("fighter", "Fighter", 10, (A.STR, A.CON), (A.STR, A.CON),
                   (ArmorCategory.LIGHT, ArmorCategory.MEDIUM, ArmorCategory.HEAVY, ArmorCategory.SHIELD),
                   ("simple", "martial"), features_by_level={1: ("second_wind",), 5: ("extra_attack",)},
                   resources=(ResourceDef("second_wind", "Second Wind", (0, 1, 1, 1, 1, 1), "short"),))
ROGUE = ClassDef("rogue", "Rogue", 8, (A.DEX,), (A.DEX, A.INT), (ArmorCategory.LIGHT,), ("simple", "shortsword",
                 "thieves_tools"), features_by_level={1: ("sneak_attack",), 5: ("uncanny_dodge",)})
CLERIC = ClassDef("cleric", "Cleric", 8, (A.WIS,), (A.WIS, A.CHA), (ArmorCategory.LIGHT, ArmorCategory.MEDIUM,
                  ArmorCategory.SHIELD), ("simple",),
                  spellcasting=SpellcastingProgression(A.WIS, "full", True, (0, 3, 3, 3, 4, 4), (), FULL_SLOTS))
WIZARD = ClassDef("wizard", "Wizard", 6, (A.INT,), (A.INT, A.WIS), (), ("dagger", "quarterstaff"),
                  spellcasting=SpellcastingProgression(A.INT, "full", True, (0, 3, 3, 3, 4, 4), (), FULL_SLOTS))
HUMAN = RaceDef("human", "Human", ability_bonuses={a: 1 for a in A})
DWARF = RaceDef("hill_dwarf", "Hill Dwarf", Size.MEDIUM, 25, {A.CON: 2, A.WIS: 1}, 60, ("dwarven_resilience",),
                resistances=(DT.POISON,), hp_per_level=1)
HALFLING = RaceDef("halfling", "Halfling", Size.SMALL, 25, {A.DEX: 2}, traits=("lucky",))


def register_all() -> None:
    F.clear_defs()
    F.register_defs(*WEAPONS, *ARMORS, *ITEMS, *SPELLS, *MONSTERS, *FEATS, FIGHTER, ROGUE, CLERIC, WIZARD, HUMAN,
                    DWARF, HALFLING)


register_all()


def build_of(cls: str, level: int = 1, race="human", **kw) -> CharacterBuild:
    base = {
        "fighter": dict(base_abilities=(15, 13, 14, 8, 10, 12), weapons=("longsword", "handaxe"), armor="chain_mail",
                        shield=True, items=("potion_of_healing", "torch"), skills=(Skill.ATHLETICS, Skill.PERCEPTION)),
        "rogue": dict(base_abilities=(8, 15, 13, 12, 14, 10), weapons=("shortsword", "dagger", "shortbow"),
                      armor="leather", items=("thieves_tools", "potion_of_healing", "torch"),
                      skills=(Skill.STEALTH, Skill.PERCEPTION, Skill.ACROBATICS)),
        "cleric": dict(base_abilities=(14, 10, 14, 8, 15, 12), weapons=("mace",), armor="chain_shirt", shield=True,
                       cantrips=(), spells=("cure_wounds", "healing_word", "bless", "hold_person", "spirit_guardians"),
                       items=("potion_of_healing", "torch")),
        "wizard": dict(base_abilities=(8, 14, 13, 15, 12, 10), weapons=("quarterstaff", "dagger"),
                       cantrips=("fire_bolt",), spells=("magic_missile", "shield", "fireball", "web", "misty_step",
                                                        "counterspell", "burning_hands", "fog_cloud"), items=("torch",)),
    }[cls]
    d = dict(name=cls.capitalize(), race=race, class_id=cls, level=level, **base)
    d.update(kw)
    return CharacterBuild(**d)


# ------------------------------------------------------------------------------------------ maps & worlds
def make_map(w: int = 20, h: int = 20, walls=(), terrain=None, doors=(), objects=(), lights=(), ambient="bright",
             regions=None) -> MapData:
    t = np.full((h, w), int(Terrain.FLOOR), dtype=np.uint8)
    t[0, :] = t[-1, :] = int(Terrain.WALL)
    t[:, 0] = t[:, -1] = int(Terrain.WALL)
    for (x, y) in walls:
        t[y, x] = int(Terrain.WALL)
    for (x, y), v in (terrain or {}).items():
        t[y, x] = int(v)
    m = np.full((h, w), int(Material.STONE_FLOOR), dtype=np.uint8)
    return MapData(w, h, t, m, list(doors), list(objects), list(lights), ambient=ambient, regions=regions or {})


def make_world(md: MapData | None = None, seed: int = 1, rules: dict | None = None, start: bool = False,
               hostile: bool = True) -> World:
    register_all()
    md = md or make_map()
    r = RulesConfig.from_overrides(rules)
    w = World(md, r, seed)
    w.add_faction("party", "Party", (200, 170, 90))
    w.add_faction("monsters", "Monsters", (200, 40, 40))
    if not hostile:
        w.set_relation(0, 1, Relation.NEUTRAL)
    if start:
        w.start()
    return w


def add_pc(w: World, cls="fighter", x=3, y=3, level=1, faction=0, controller=ControllerKind.SCRIPTED, light=False,
           **kw) -> int:
    b = build_of(cls, level, **kw)
    c = F.make_pc(b, w.rng, w.rules.pc_hp)
    c.x, c.y, c.faction, c.controller = x, y, faction, controller
    return w.spawn_creature(c, light=light)


def add_mon(w: World, md="goblin", x=8, y=8, faction=1, controller=ControllerKind.SCRIPTED, name=None) -> int:
    d = F.lookup_monster(md) if isinstance(md, str) else md
    c = F.make_monster(d, w.rng, w.rules.monster_hp, name)
    c.x, c.y, c.faction, c.controller = x, y, faction, controller
    return w.spawn_creature(c)


def ab_index(w: World, eid: int, ability_id: str) -> int:
    for i, a in enumerate(w.abilities(eid)):
        if a.id == ability_id:
            return i
    raise KeyError(ability_id)


def do(w: World, eid: int, ability_id: str, target: int = -1, point=None) -> None:
    """Issue ABILITY act for ``eid`` (must be the decider)."""
    assert w.decider() == eid, f"decider is {w.decider()}, not {eid}"
    w.step(Act(ActKind.ABILITY, ab_index(w, eid, ability_id), target, point))


def to_turn(w: World, eid: int, limit: int = 200) -> None:
    """End turns until ``eid`` is the active creature."""
    n = 0
    while w.decider() != eid and not w.done and n < limit:
        w.step(Act(ActKind.END_TURN))
        n += 1
    assert w.decider() == eid


def events_of(w: World, *types):
    ev = w.drain_events()
    return [e for e in ev if not types or e.t in [int(t) for t in types]]


def set_active(w: World, eid: int) -> None:
    """Force ``eid`` to be the active creature with a fresh turn (test helper)."""
    c = w.entities[eid]
    w.active = eid
    c.ts.reset(c.speeds.get("walk", 30))
    c.reaction_available = True
    if eid in w.order:
        w.turn_idx = w.order.index(eid)


# ------------------------------------------------------------------------------------------ random policy
def random_act(w: World, rng: np.random.Generator, max_ent: int = 16) -> Act:
    """A random *legal* act for ``w.decider()`` built from the engine masks (used by fuzz tests and the benchmark)."""
    d = w.decider()
    if w.pending_decision is not None:
        n = len(w.pending_decision.options)
        return Act(ActKind.DECIDE, int(rng.integers(n)))
    c = w.entities[d]
    cands = [e.eid for e in w.entities if (e.kind == 0 and e.present) or (e.kind != 0 and e.present)][:max_ent]
    if d not in cands:
        cands = [d] + cands[:max_ent - 1]
    cand = np.array(cands, dtype=np.int64)
    pts = np.array([[c.x + dx, c.y + dy] for dx in (-3, 0, 3) for dy in (-3, 0, 3)], dtype=np.int64)
    am = w.ability_mask(d)
    mm = w.move_mask(d)
    options = [("end", None)]
    if mm.any():
        options.append(("move", np.nonzero(mm)[0]))
    ai = np.nonzero(am)[0]
    if len(ai):
        options.append(("ab", ai))
    ap = w.approach_mask(d, cand)
    if ap.any():
        options.append(("app", np.nonzero(ap)[0]))
    kind, arr = options[int(rng.integers(len(options)))]
    if kind == "end":
        if rng.random() < 0.7 and len(options) > 1:
            kind, arr = options[1 + int(rng.integers(len(options) - 1))]
        else:
            return Act(ActKind.END_TURN)
    if kind == "move":
        k = int(arr[int(rng.integers(len(arr)))])
        if k == 8:
            return Act(ActKind.ASCEND)
        if k == 9:
            return Act(ActKind.DESCEND)
        return Act(ActKind.MOVE, k)
    if kind == "app":
        return Act(ActKind.APPROACH, target=int(cand[int(arr[int(rng.integers(len(arr)))])]))
    k = int(arr[int(rng.integers(len(arr)))])
    tm = w.target_mask(d, k, cand, pts)
    ti = np.nonzero(tm)[0]
    ab = w.abilities(d)[k]
    if len(ti) == 0:
        return Act(ActKind.END_TURN)
    t = int(ti[int(rng.integers(len(ti)))])
    if t < len(cand):
        return Act(ActKind.ABILITY, k, int(cand[t]), None)
    p = pts[t - len(cand)]
    return Act(ActKind.ABILITY, k, -1, (int(p[0]), int(p[1])))


def skirmish(seed: int = 0, n_pc: int = 4, n_gob: int = 6, size: int = 30, rules: dict | None = None) -> World:
    """4 PCs vs 6 goblins on a 30x30 map (the benchmark scenario)."""
    classes = ["fighter", "cleric", "rogue", "wizard"]
    walls = [(10, y) for y in range(5, 12)] + [(x, 18) for x in range(12, 20)]
    walls = [(x, y) for (x, y) in walls if x < size - 2 and y < size - 2]
    terr = {p: Terrain.DIFFICULT for p in [(15, 15), (16, 15)] if p[0] < size - 2 and p[1] < size - 2}
    md = make_map(size, size, walls=walls, ambient="bright", terrain=terr)
    w = make_world(md, seed, rules)
    for i in range(n_pc):
        add_pc(w, classes[i % 4], min(3 + i, size - 2), 3, level=3, controller=ControllerKind.AGENT)
    for i in range(n_gob):
        add_mon(w, "goblin", min(12 + i % 3, size - 2), min(10 + i // 3, size - 2))
    w.start()
    return w


class ScriptedRng:
    """Deterministic stand-in for ``world.rng``: ``random()`` yields queued fractions (use ``face(n, sides)``),
    then ``default`` forever.  Only for unit tests that need exact dice."""

    def __init__(self, faces=(), default: float = 0.5):
        self.q = list(faces)
        self.default = default

    def random(self):
        return self.q.pop(0) if self.q else self.default

    def integers(self, *a, **k):
        return 0


def face(n: int, sides: int = 20) -> float:
    """Fraction that makes ``int(r * sides) + 1 == n``."""
    return (n - 0.5) / sides


def d20s(*ns, default_face=10):
    """ScriptedRng returning the given d20 faces first (further dice at ``default_face`` of a d20-ish fraction)."""
    return ScriptedRng([face(n) for n in ns], face(default_face))
