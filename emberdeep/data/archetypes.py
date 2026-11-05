"""Canonical, tactically sensible character builds for levels 1-5 (and random builds).

``archetype(class_id, level, race)`` returns a :class:`~emberdeep.data.schema.CharacterBuild` with exact SRD
spell counts, features, equipment and ability scores (standard array assigned to primary abilities, racial
bonuses applied later by the factory via :func:`final_abilities`).

Build conventions (consumed by ``core/factory.py``):

* ``build.cantrips`` / ``build.spells`` - class cantrips and known (or prepared) leveled spells. Always-prepared
  subclass spells (life, devotion, land) and racial / pact-tome cantrips are NOT included.
* ``build.choices`` keys: ``racial_cantrip`` (high elf), ``racial_skills`` (half-elf), ``half_elf_bonus``,
  ``draconic_ancestry``, ``land`` (druid), ``favored_enemy`` / ``favored_terrain`` / ``hunters_prey`` (ranger),
  ``metamagic``, ``invocations``, ``pact_boon``, ``tome_cantrips``, ``spellbook`` (wizard), ``lore_skills``,
  ``land_cantrip`` (druid circle bonus cantrip, already counted in ``cantrips``).
* ``build.skills`` holds the class skill picks only.
"""
from __future__ import annotations

from typing import Any, Callable

from emberdeep.core.types import Ability as A
from emberdeep.core.types import ArmorCategory, Skill as S
from emberdeep.data.armor import ARMOR
from emberdeep.data.classes import (
    CLASSES, CLASS_SPELL_LISTS, ALWAYS_PREPARED_SUBCLASSES, bonus_spells_up_to, spell_counts,
)
from emberdeep.data.races import DRAGON_ANCESTRY, HIGH_ELF_CANTRIP_OPTIONS, RACES
from emberdeep.data.schema import CharacterBuild
from emberdeep.data.spell_ids import SPELL_LEVEL
from emberdeep.data.weapons import WEAPONS
from emberdeep.core.types import WeaponProperty

__all__ = ["archetype", "ARCHETYPES", "random_build", "final_abilities", "NAME_POOL", "VARIANT_SPECS",
           "DEFAULT_VARIANT", "roll_abilities"]

NAME_POOL: tuple[str, ...] = (
    "Aldric Vane", "Brannoch Hale", "Cassia Thornfield", "Dorrin Ashgrove", "Elowen Marsh", "Fenwick Dray",
    "Galena Voss", "Hadrin Coalbright", "Isolde Quillon", "Jorund Brackenridge", "Kestrel Wynn", "Lysandra Pike",
    "Maelis Corrow", "Nerys Holloway", "Orsolo Venn", "Perrin Stoneweather", "Quillan Ashby", "Rhosyn Tallow",
    "Sorrel Dunmere", "Tamsin Greaves", "Ulric Fenwater", "Vesper Locke", "Wynstan Ridley", "Xanthe Marrow",
    "Yorick Salt", "Zephra Lindqvist", "Ansel Rook", "Bryn Tolliver", "Corwin Maddox", "Delphine Orrow",
    "Evander Cray", "Fiora Nettlebed", "Gideon Harrowgate", "Hesper Wick", "Idris Calloway", "Juniper Skerry",
    "Kellan Frost", "Lorcan Dree", "Mirelle Ashdown", "Nikolai Brandt", "Odalys Penhallow", "Peregrine Moss",
    "Rowan Eldergast", "Sabine Thorne", "Torren Gale", "Una Whitlock", "Valen Corbin", "Willa Fairweather",
    "Xavier Pell", "Ysolde Marchbanks", "Zora Kettleby", "Alaric Dunstan", "Briar Lannoc", "Caden Ostrow",
    "Dara Quillfeather", "Emeric Holt", "Fallon Rue", "Garrick Sable", "Hollis Tarn", "Ione Brightwater",
    "Jasper Coldwell", "Keira Mossgrave", "Lucan Vey", "Marisol Drake", "Nolan Ironsides", "Oriel Wren",
    "Petra Hollin", "Quentin Larch", "Ravenna Slate", "Silas Thistledown", "Thea Gravenor", "Ulla Ferrow",
)

_SMALL_SWAP = {"greatsword": "longsword", "greataxe": "battleaxe", "longbow": "shortbow", "glaive": "spear",
               "halberd": "spear", "heavy_crossbow": "light_crossbow", "maul": "warhammer", "pike": "spear"}

# ---------------------------------------------------------------------------------------------
# Variant specifications
# ---------------------------------------------------------------------------------------------
# base: (STR, DEX, CON, INT, WIS, CHA) standard-array assignment before racial bonuses.
VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "fighter": dict(
        class_id="fighter", race="human", base=(15, 13, 14, 8, 12, 10), asi=A.STR, style="dueling",
        skills=(S.ATHLETICS, S.PERCEPTION), weapons=("longsword", "handaxe"), armor="chain_mail", shield=True,
        items=("potion_of_healing",)),
    "fighter_greatsword": dict(
        class_id="fighter", race="half_orc", base=(15, 13, 14, 8, 12, 10), asi=A.STR,
        style="great_weapon_fighting", skills=(S.ATHLETICS, S.INTIMIDATION), weapons=("greatsword", "handaxe"),
        armor="chain_mail", shield=False, items=("potion_of_healing",)),
    "fighter_archer": dict(
        class_id="fighter", race="human", base=(13, 15, 14, 8, 12, 10), asi=A.DEX, style="archery",
        skills=(S.ACROBATICS, S.PERCEPTION), weapons=("longbow", "shortsword"), armor="chain_mail", shield=False,
        items=("potion_of_healing", "arrows_20")),
    "barbarian": dict(
        class_id="barbarian", race="half_orc", base=(15, 13, 14, 8, 12, 10), asi=A.STR,
        skills=(S.ATHLETICS, S.PERCEPTION), weapons=("greataxe", "handaxe", "javelin"), armor="", shield=False,
        items=("potion_of_healing",)),
    "bard": dict(
        class_id="bard", race="half_elf", base=(8, 14, 13, 10, 12, 15), asi=A.CHA,
        skills=(S.PERSUASION, S.DECEPTION, S.PERCEPTION), weapons=("rapier", "dagger"), armor="leather",
        shield=False, items=("potion_of_healing", "musical_instrument"),
        cantrips=("vicious_mockery", "minor_illusion", "dancing_lights", "light", "mage_hand", "true_strike"),
        spells=("healing_word", "faerie_fire", "sleep", "thunderwave", "cure_wounds", "hold_person", "invisibility",
                "hypnotic_pattern", "dispel_magic", "heroism", "bane", "charm_person"),
        expertise=(S.PERSUASION, S.DECEPTION), lore_skills=(S.ARCANA, S.HISTORY, S.INVESTIGATION),
        racial_skills=(S.INSIGHT, S.STEALTH), bonus=(A.DEX, A.CON)),
    "cleric": dict(
        class_id="cleric", race="hill_dwarf", base=(13, 10, 14, 8, 15, 12), asi=A.WIS,
        skills=(S.MEDICINE, S.RELIGION), weapons=("mace", "light_crossbow"), armor="chain_mail", shield=True,
        items=("potion_of_healing", "holy_symbol", "bolts_20"),
        cantrips=("sacred_flame", "guidance", "spare_the_dying", "thaumaturgy", "light"),
        spells=("healing_word", "guiding_bolt", "shield_of_faith", "sanctuary", "command", "spirit_guardians",
                "hold_person", "aid", "silence", "dispel_magic", "mass_healing_word", "prayer_of_healing",
                "bane", "inflict_wounds")),
    "druid": dict(
        class_id="druid", race="human", base=(8, 13, 14, 12, 15, 10), asi=A.WIS,
        skills=(S.PERCEPTION, S.SURVIVAL), weapons=("scimitar", "sling"), armor="hide", shield=True,
        items=("potion_of_healing", "druidic_focus", "sling_bullets_20"), land="forest",
        cantrips=("produce_flame", "shillelagh", "guidance", "poison_spray", "resistance"),
        land_cantrip="guidance",
        spells=("entangle", "healing_word", "faerie_fire", "goodberry", "moonbeam", "flaming_sphere",
                "conjure_animals", "cure_wounds", "thunderwave", "hold_person", "lesser_restoration",
                "dispel_magic", "sleet_storm", "barkskin")),
    "monk": dict(
        class_id="monk", race="human", base=(12, 15, 13, 8, 14, 10), asi=A.DEX,
        skills=(S.ACROBATICS, S.STEALTH), weapons=("shortsword", "dart"), armor="", shield=False,
        items=("potion_of_healing",)),
    "paladin": dict(
        class_id="paladin", race="dragonborn", base=(15, 10, 13, 8, 12, 14), asi=A.STR, style="dueling",
        skills=(S.ATHLETICS, S.PERSUASION), weapons=("longsword", "javelin"), armor="chain_mail", shield=True,
        items=("potion_of_healing", "holy_symbol"), ancestry="gold",
        spells=("bless", "cure_wounds", "shield_of_faith", "command", "divine_favor", "heroism", "aid",
                "magic_weapon", "protection_from_evil_and_good", "lesser_restoration")),
    "ranger": dict(
        class_id="ranger", race="high_elf", base=(12, 15, 13, 10, 14, 8), asi=A.DEX, style="archery",
        skills=(S.STEALTH, S.SURVIVAL, S.NATURE), weapons=("longbow", "shortsword"), armor="leather", shield=False,
        items=("potion_of_healing", "arrows_20"), favored_enemy="undead", favored_terrain="forest",
        hunters_prey="colossus_slayer", racial_cantrip="fire_bolt",
        spells=("hunters_mark", "cure_wounds", "goodberry", "pass_without_trace", "spike_growth", "longstrider",
                "fog_cloud", "jump")),
    "rogue": dict(
        class_id="rogue", race="lightfoot_halfling", base=(8, 15, 14, 10, 12, 13), asi=A.DEX,
        skills=(S.STEALTH, S.PERCEPTION, S.ACROBATICS, S.SLEIGHT_OF_HAND), weapons=("rapier", "shortbow", "dagger"),
        armor="leather", shield=False, items=("potion_of_healing", "thieves_tools", "arrows_20"),
        expertise=(S.STEALTH, S.PERCEPTION)),
    "sorcerer": dict(
        class_id="sorcerer", race="tiefling", base=(8, 13, 14, 10, 12, 15), asi=A.CHA,
        skills=(S.PERSUASION, S.ARCANA), weapons=("dagger", "light_crossbow"), armor="", shield=False,
        items=("potion_of_healing", "arcane_focus", "bolts_20"), ancestry="red",
        cantrips=("fire_bolt", "ray_of_frost", "mage_hand", "minor_illusion", "shocking_grasp", "light"),
        spells=("magic_missile", "shield", "sleep", "misty_step", "scorching_ray", "fireball", "mage_armor",
                "burning_hands", "haste", "counterspell"),
        metamagic=("quickened_spell", "twinned_spell", "empowered_spell", "distant_spell")),
    "warlock": dict(
        class_id="warlock", race="tiefling", base=(8, 13, 14, 10, 12, 15), asi=A.CHA,
        skills=(S.ARCANA, S.DECEPTION), weapons=("dagger", "light_crossbow"), armor="leather", shield=False,
        items=("potion_of_healing", "arcane_focus", "bolts_20"),
        cantrips=("eldritch_blast", "chill_touch", "mage_hand", "minor_illusion", "poison_spray"),
        spells=("hellish_rebuke", "charm_person", "protection_from_evil_and_good", "misty_step", "hold_person",
                "hypnotic_pattern", "counterspell", "fear", "darkness", "expeditious_retreat"),
        invocations=("agonizing_blast", "armor_of_shadows", "devils_sight", "repelling_blast", "fiendish_vigor"),
        pact_boon="pact_of_the_tome", tome_cantrips=("guidance", "spare_the_dying", "sacred_flame")),
    "wizard": dict(
        class_id="wizard", race="high_elf", base=(8, 14, 13, 15, 12, 10), asi=A.INT,
        skills=(S.ARCANA, S.INVESTIGATION), weapons=("quarterstaff", "dagger"), armor="", shield=False,
        items=("potion_of_healing", "spellbook", "arcane_focus"), racial_cantrip="minor_illusion",
        cantrips=("fire_bolt", "ray_of_frost", "mage_hand", "shocking_grasp", "light", "true_strike"),
        spells=("magic_missile", "shield", "mage_armor", "sleep", "scorching_ray", "misty_step", "fireball", "web",
                "burning_hands", "counterspell", "thunderwave", "mirror_image", "haste"),
        book_extra=("detect_magic", "feather_fall", "grease", "fog_cloud", "invisibility", "hold_person",
                    "acid_arrow", "fly", "dispel_magic", "slow", "blur", "expeditious_retreat", "charm_person")),
}
DEFAULT_VARIANT = {c: c for c in CLASSES}
DEFAULT_VARIANT["fighter"] = "fighter"

_TINT = {
    "barbarian": (190, 80, 50), "bard": (200, 120, 190), "cleric": (235, 225, 170), "druid": (110, 170, 90),
    "fighter": (170, 170, 185), "monk": (220, 160, 80), "paladin": (240, 210, 110), "ranger": (90, 140, 80),
    "rogue": (90, 90, 110), "sorcerer": (220, 70, 70), "warlock": (130, 70, 170), "wizard": (80, 120, 220),
}
_FIGHTER_VARIANTS = ("fighter", "fighter_greatsword", "fighter_archer")


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def final_abilities(build: CharacterBuild) -> tuple[int, int, int, int, int, int]:
    """Final ability scores: base + racial bonuses (+ half-elf choice) + ASI entries, capped at 20."""
    race = RACES[build.race]
    scores = list(build.base_abilities)
    for ab, bonus in race.ability_bonuses.items():
        scores[int(ab)] += bonus
    for ab in build.choices.get("half_elf_bonus", ()):
        scores[int(ab)] += 1
    if build.level >= 4:
        for ab in build.asi:
            scores[int(ab)] = min(20, scores[int(ab)] + 1)
    return tuple(scores)  # type: ignore[return-value]


def roll_abilities(rng: Any) -> list[int]:
    """Roll six scores with 4d6-drop-lowest, sorted descending."""
    out = []
    for _ in range(6):
        d = sorted(int(x) for x in rng.integers(1, 7, size=4))
        out.append(sum(d[1:]))
    return sorted(out, reverse=True)


def _ranking(base: tuple[int, ...]) -> list[int]:
    """Ability indices ordered from highest to lowest base score (ties: earlier index first)."""
    return sorted(range(6), key=lambda i: (-base[i], i))


def _racial_skills(race_id: str) -> tuple[S, ...]:
    return RACES[race_id].skill_proficiencies


def _pick_skills(want: tuple[S, ...], cls_id: str, taken: set[S]) -> tuple[S, ...]:
    """Class skill picks: preferred skills first, filling from the class list; avoid ``taken`` duplicates."""
    n, options = CLASSES[cls_id].skill_choices
    out: list[S] = []
    for s in tuple(want) + tuple(options):
        if s in options and s not in out and s not in taken:
            out.append(s)
        if len(out) == n:
            break
    return tuple(out)


def _legal_spell(cls_id: str, sid: str, msl: int, level_exact: int | None = None) -> bool:
    return sid in CLASS_SPELL_LISTS[cls_id] and 1 <= SPELL_LEVEL[sid] <= msl


def _select(cls_id: str, prefs: tuple[str, ...], n: int, msl: int, exclude: set[str]) -> tuple[str, ...]:
    out: list[str] = []
    for sid in tuple(prefs) + CLASS_SPELL_LISTS[cls_id]:
        if len(out) == n:
            break
        if sid in out or sid in exclude or not _legal_spell(cls_id, sid, msl):
            continue
        out.append(sid)
    return tuple(out)


def _select_cantrips(cls_id: str, prefs: tuple[str, ...], n: int, exclude: set[str]) -> tuple[str, ...]:
    out: list[str] = []
    for sid in tuple(prefs) + CLASS_SPELL_LISTS[cls_id]:
        if len(out) == n:
            break
        if sid in out or sid in exclude or sid not in CLASS_SPELL_LISTS[cls_id] or SPELL_LEVEL[sid] != 0:
            continue
        out.append(sid)
    return tuple(out)


def _fit_armor(cls_id: str, armor: str, shield: bool, final_str: int, sub: str = "") -> tuple[str, bool]:
    """Downgrade armor the character cannot wear (proficiency / strength requirement)."""
    prof = set(CLASSES[cls_id].armor_proficiencies)
    if sub == "life":
        prof.add(ArmorCategory.HEAVY)   # Life Domain bonus proficiency
    shield = shield and ArmorCategory.SHIELD in prof
    if not armor:
        return "", shield
    order = ("plate", "splint", "chain_mail", "ring_mail", "half_plate", "breastplate", "scale_mail", "chain_shirt",
             "hide", "studded_leather", "leather", "padded")
    for a in (armor,) + order:
        d = ARMOR[a]
        if d.category in prof and d.str_req <= final_str:
            return a, shield
    return "", shield


def _small_weapons(weapons: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for w in weapons:
        w = _SMALL_SWAP.get(w, w) if WEAPONS[w].properties & WeaponProperty.HEAVY else w
        if w not in out:
            out.append(w)
    return tuple(out)


def _asi_entries(scores: list[int], primary: A, build_scores_rank: list[int]) -> tuple[A, ...]:
    p = int(primary)
    if scores[p] + 2 <= 20:
        return (A(p), A(p))
    if scores[p] + 1 <= 20:
        for i in build_scores_rank:
            if i != p and scores[i] + 1 <= 20:
                return (A(p), A(i))
    for i in build_scores_rank:
        if scores[i] + 2 <= 20:
            return (A(i), A(i))
    return ()


# ---------------------------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------------------------
def build_variant(variant: str, level: int = 1, race: str | None = None, name: str | None = None,
                  base_abilities: tuple[int, ...] | None = None, rng: Any = None) -> CharacterBuild:
    """Build a :class:`CharacterBuild` for a named variant (see :data:`VARIANT_SPECS`)."""
    if not 1 <= level <= 5:
        raise ValueError("archetypes support levels 1-5")
    spec = VARIANT_SPECS[variant]
    cls_id: str = spec["class_id"]
    cdef = CLASSES[cls_id]
    race_id = race or spec["race"]
    rdef = RACES[race_id]
    base = tuple(base_abilities) if base_abilities is not None else tuple(spec["base"])
    choices: dict[str, Any] = {}

    # --- race-specific choices
    if race_id == "dragonborn":
        anc = spec.get("ancestry")
        if anc is None or rng is not None:
            keys = sorted(DRAGON_ANCESTRY)
            anc = keys[int(rng.integers(0, len(keys)))] if rng is not None else "red"
        choices["draconic_ancestry"] = anc
    rank = _ranking(tuple(spec["base"]))
    if race_id == "half_elf":
        bonus = spec.get("bonus")
        if bonus is None or race != spec["race"]:
            bonus = tuple(A(i) for i in rank if i != int(A.CHA))[:2]
        choices["half_elf_bonus"] = tuple(bonus)
    # Standard-array assignment: scores sorted by the variant's ranking when custom abilities are supplied.
    if base_abilities is not None:
        ordered = sorted(base, reverse=True)
        arr = [0] * 6
        for pos, idx in enumerate(rank):
            arr[idx] = ordered[pos]
        base = tuple(arr)

    # --- ability scores (before ASI)
    scores = list(base)
    for ab, b in rdef.ability_bonuses.items():
        scores[int(ab)] += b
    for ab in choices.get("half_elf_bonus", ()):
        scores[int(ab)] += 1
    asi: tuple[A, ...] = ()
    if level >= 4:
        asi = _asi_entries(scores, spec["asi"], rank)
        for ab in asi:
            scores[int(ab)] = min(20, scores[int(ab)] + 1)
    final = tuple(scores)

    # --- subclass
    sub = cdef.subclasses[0] if level >= cdef.subclass_level else ""

    if race_id == "high_elf":
        rc = spec.get("racial_cantrip", "fire_bolt")
        choices["racial_cantrip"] = rc if rc in HIGH_ELF_CANTRIP_OPTIONS else "fire_bolt"

    # --- skills
    taken = set(_racial_skills(race_id))
    skills = _pick_skills(tuple(spec["skills"]), cls_id, taken)
    expertise: tuple[S, ...] = ()
    if cls_id == "rogue":
        expertise = tuple(s for s in spec["expertise"] if s in skills) or skills[:2]
    if cls_id == "bard" and level >= 3:
        expertise = tuple(s for s in spec["expertise"] if s in skills) or skills[:2]
        choices["lore_skills"] = tuple(spec["lore_skills"])
    if race_id == "half_elf":
        rs = tuple(s for s in spec.get("racial_skills", (S.INSIGHT, S.STEALTH)) if s not in skills)
        extra = [s for s in S if s not in skills and s not in rs and s not in taken]
        rs = (rs + tuple(extra))[:2]
        choices["racial_skills"] = rs

    # --- spells
    cantrips: tuple[str, ...] = ()
    spells: tuple[str, ...] = ()
    counts = spell_counts(cls_id, level, final)
    exclude: set[str] = set()
    if cdef.spellcasting is not None:
        msl = counts.max_spell_level
        always = set(bonus_spells_up_to(sub, level, spec.get("land", ""))) if sub in ALWAYS_PREPARED_SUBCLASSES else set()
        exclude |= always
        racial_c = set(rdef.cantrips)
        if race_id == "high_elf":
            racial_c.add(choices["racial_cantrip"])
        n_c = counts.cantrips
        land_c = ""
        if cls_id == "druid" and level >= 2:
            land_c = spec["land_cantrip"]
        prefs_c = tuple(c for c in spec.get("cantrips", ()) if c not in racial_c)
        cantrips = _select_cantrips(cls_id, prefs_c, n_c, racial_c | ({land_c} if land_c else set()))
        if land_c:
            cantrips = cantrips + (land_c,)
            choices["land_cantrip"] = land_c
        if n_c:
            # class cantrips must not collide with racial cantrips of the same name
            assert not (set(cantrips) & racial_c)
        if counts.leveled:
            spells = _select(cls_id, tuple(spec.get("spells", ())), counts.leveled, msl, exclude)
        if cls_id == "wizard":
            extras = tuple(spec.get("book_extra", ()))
            book = list(spells)
            for sid in extras + CLASS_SPELL_LISTS["wizard"]:
                if len(book) >= counts.spellbook:
                    break
                if sid not in book and _legal_spell("wizard", sid, msl):
                    book.append(sid)
            choices["spellbook"] = tuple(book)

    # --- class-specific choices
    if cls_id == "druid":
        choices["land"] = spec["land"]
    if cls_id == "ranger":
        choices["favored_enemy"] = spec["favored_enemy"]
        choices["favored_terrain"] = spec["favored_terrain"]
        if level >= 3:
            choices["hunters_prey"] = spec["hunters_prey"]
    if cls_id == "sorcerer":
        choices["draconic_ancestry"] = spec["ancestry"] if rng is None else sorted(DRAGON_ANCESTRY)[
            int(rng.integers(0, len(DRAGON_ANCESTRY)))]
        if level >= 3:
            choices["metamagic"] = tuple(spec["metamagic"][:2])
    if cls_id == "warlock":
        if level >= 2:
            n_inv = 2 if level < 5 else 3
            choices["invocations"] = tuple(spec["invocations"][:n_inv])
        if level >= 3:
            choices["pact_boon"] = spec["pact_boon"]
            if spec["pact_boon"] == "pact_of_the_tome":
                choices["tome_cantrips"] = tuple(spec["tome_cantrips"])
    fighting_style = ""
    if "style" in spec and (cls_id == "fighter" or level >= 2):
        fighting_style = spec["style"]

    # --- equipment
    armor, shield = spec["armor"], spec["shield"]
    if cls_id == "warlock" and level >= 2:
        armor = ""   # armor of shadows
    armor, shield = _fit_armor(cls_id, armor, shield, final[int(A.STR)], sub)
    weapons = tuple(spec["weapons"])
    if rdef.size.value <= 1:   # Small: swap heavy weapons
        weapons = _small_weapons(weapons)
    items = list(spec["items"])
    if rdef.darkvision_ft == 0:
        items.append("torch")
    gold = 10 + 15 * (level - 1)

    if name is None:
        key = (list(VARIANT_SPECS).index(variant) * 11 + list(RACES).index(race_id) * 5 + level) % len(NAME_POOL)
        name = NAME_POOL[key]
    return CharacterBuild(
        name=name, race=race_id, class_id=cls_id, level=level, subclass=sub, base_abilities=base, asi=asi,
        skills=skills, expertise=expertise, fighting_style=fighting_style, cantrips=cantrips, spells=spells,
        choices=choices, weapons=weapons, armor=armor, shield=shield, items=tuple(items), gold=gold,
        icon=cdef.icon, portrait_tint=_TINT[cls_id])


def archetype(class_id: str, level: int = 1, race: str | None = None, name: str | None = None) -> CharacterBuild:
    """Canonical build for ``class_id`` (or a variant name) at ``level`` (1-5), optionally overriding the race."""
    if class_id in VARIANT_SPECS:     # named variants ("fighter_archer") are accepted too
        return build_variant(class_id, level, race, name)
    if class_id not in CLASSES:
        raise KeyError(class_id)
    return build_variant(DEFAULT_VARIANT[class_id], level, race, name)


def _make_archetype(variant: str) -> Callable[..., CharacterBuild]:
    def make(level: int = 1, race: str | None = None, name: str | None = None) -> CharacterBuild:
        return build_variant(variant, level, race, name)
    make.__name__ = f"archetype_{variant}"
    make.__doc__ = f"Build the '{variant}' archetype at the given level."
    return make


ARCHETYPES: dict[str, Callable[..., CharacterBuild]] = {v: _make_archetype(v) for v in VARIANT_SPECS}


def random_build(rng: Any, level: int, class_id: str | None = None, race: str | None = None,
                 roll_scores: bool = True) -> CharacterBuild:
    """Randomised legal build for domain randomisation.

    Race and class are drawn from ``rng`` unless given; ability scores are 4d6-drop-lowest (or the standard
    array when ``roll_scores`` is false) assigned to the variant's priorities; ancestry / names vary.
    """
    cid = class_id or sorted(CLASSES)[int(rng.integers(0, len(CLASSES)))]
    if cid == "fighter":
        variant = _FIGHTER_VARIANTS[int(rng.integers(0, len(_FIGHTER_VARIANTS)))]
    else:
        variant = DEFAULT_VARIANT[cid]
    race_ids = sorted(RACES)
    rid = race or race_ids[int(rng.integers(0, len(race_ids)))]
    scores = tuple(roll_abilities(rng)) if roll_scores else (15, 14, 13, 12, 10, 8)
    name = NAME_POOL[int(rng.integers(0, len(NAME_POOL)))]
    return build_variant(variant, level, rid, name, base_abilities=scores, rng=rng)
