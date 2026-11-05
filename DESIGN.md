# Emberdeep — Architecture & Design Specification

Emberdeep is a multi-agent reinforcement-learning environment that implements tactical
fantasy role-playing combat and dungeon exploration using the **5e System Reference
Document 5.1 (CC-BY-4.0)** ruleset. It targets research on **cooperation, social
dilemmas and general-sum games**: parties share some goals (survive, defeat monsters) and
compete over others (loot, XP, safety, who takes risks), and multiple parties can negotiate,
truce, or betray.

It has two faces:

* **Training mode**: headless, pure Python + NumPy (+ optional Numba), no graphics imports,
  deterministic, clonable state, PettingZoo AEC/Parallel and Gymnasium APIs, vectorised
  environments, CTDE global state, social metrics.
* **Render mode**: an OpenGL (moderngl) + pygame-ce presentation layer with a professional,
  polished look: dynamic lighting & shadows, normal-mapped procedural materials, particles,
  spell VFX, post-processing, animated tokens, a full HUD (initiative tracker, party frames,
  combat log with dice, tooltips), human play, replays, and video capture.

The two are connected **only** through `emberdeep/render/api.py` (plain dataclasses) and the
event stream (`emberdeep/core/events.py`).

---------------------------------------------------------------------------------------------

## 1. Principles

1. **Rules fidelity first.** Implement SRD 5.1 faithfully for character levels 1–5
   (spells up to 3rd level). Where a tabletop rule needs a judgement call, implement the
   most common table interpretation and document it in `docs/RULES_NOTES.md`.
2. **Data-driven content.** All content is declared with schemas in `emberdeep/data/schema.py`.
   Bespoke logic is attached by string `hook` ids resolved in registries (`emberdeep/rules/*`).
3. **Contracts are stable.** `core/types.py`, `core/events.py`, `data/schema.py`,
   `render/api.py`, `render/hud_api.py`, `scenarios/spec.py`, `scenarios/mapdata.py` are shared
   contracts. Append-only changes; never renumber enums.
4. **Determinism.** Every random draw goes through `world.rng` (a `numpy.random.Generator`,
   PCG64). Same seed + same actions => identical trajectories. No `random` module, no
   `hash()`-order dependence, no wall-clock.
5. **Clonable state.** `world.clone()` returns an independent copy (for MCTS/planning/inference).
   Static data (definitions, map static arrays) is shared by reference, never copied.
6. **Speed in training mode.** No string formatting in hot paths, no pygame import, cached
   vision/LOS, vectorised masks. Target ≥ 2,000 env steps / s / core on a 4v6 encounter.
7. **Python ≥ 3.10.** Use `from __future__ import annotations`; no 3.12-only syntax. Deps:
   numpy, gymnasium, pettingzoo; optional numba, pygame-ce, moderngl, torch.

---------------------------------------------------------------------------------------------

## 2. Repository layout & ownership

```
emberdeep/
  core/            engine: world state, turn flow, resolution, vision, movement (pure rules engine)
    types.py events.py          (contracts)
    dice.py                      dice expressions, rolling with world.rng
    geometry.py vision.py pathing.py   grid algorithms (numba-accelerated, pure-numpy fallback)
    entities.py                  Entity, Creature, WorldObject (Door/Chest/LootPile/Lever/Trap/Altar/Exit)
    gamemap.py                   GameMap: terrain arrays, opacity, occupancy, light grid
    abilities.py                 runtime Ability objects, universal abilities, ability list building
    resolution.py                attacks, saves, checks, damage, healing, conditions
    effects.py zones.py          ongoing effects & area zones (registry-driven)
    decisions.py                 reaction/decision machinery (NeedDecision replay)
    world.py                     World: the public engine API
    factory.py                   creatures from MonsterDef / CharacterBuild
  rules/           hook registries: features.py traits.py effects.py zones.py spells.py items.py objects.py
  data/            SRD content: weapons.py armor.py items.py races.py classes.py spells.py monsters.py archetypes.py
  scenarios/       spec.py mapdata.py ascii_map.py mapgen.py decorate.py builder.py library/*.py
  ai/              scripted policies: base.py utility.py profiles.py heroes.py registry.py
  env/             RL API: config.py aec.py parallel.py gym_single.py obs.py actions.py rewards.py metrics.py
                   vector.py wrappers.py render_adapter.py
  render/          api.py hud_api.py assets.py renderer.py gl/* vfx/* textures/* tokens.py director.py camera.py
                   ui/* audio.py mock.py
  replay.py viewer.py play.py bench.py
tests/ examples/ benchmarks/ docs/
```

---------------------------------------------------------------------------------------------

## 3. Coding conventions

* Classes holding per-episode mutable state use `__slots__` or `@dataclass(slots=True)`; they must
  be deep-copyable (no lambdas, generators, open files, or pygame objects inside world state).
* Definitions (`*Def`) are frozen and shared. Creatures reference them (`creature.monster_def`,
  `ability.source`), and `World.__deepcopy__` / `clone()` must not duplicate them (use a memo
  that maps every definition object to itself, or keep them in module-level registries
  referenced by id).
* No `print` in library code; use `logging.getLogger(__name__)`.
* Public functions have docstrings; tricky rules cite the SRD section in a comment.
* Tests live in `tests/` (pytest). Every rules module has unit tests with seeded RNG.
* Use the project venv: `/home/user/.venv/bin/python` (has numpy, numba, gymnasium,
  pettingzoo, pygame-ce, moderngl, pytest). For GL work use `DISPLAY=:99` (Xvfb running).

---------------------------------------------------------------------------------------------

## 4. Engine

### 4.1 Entities and ids

`World.entities: list[Entity]` holds every creature and world object ever created in the
episode; the list index is the entity id **`eid`** (stable, never reused). Objects removed from
play stay in the list with `present = False`.

```
Entity            eid, kind: EntityKind, name, x, y, present: bool
 ├─ Creature      (see 4.2)
 └─ WorldObject   Door | Chest | LootPile | Lever | Trap | Altar | ExitMarker
```

* `Door`: state (DoorState), lock_dc, hp, ac=15, key_id. Closed/locked doors are opaque and block movement.
* `Chest`: closed/open/locked, lock_dc, gold, items, trap id. Opening moves contents to the opener.
* `LootPile`: gold + items on a square (dropped by the dead, or placed). Picking up takes everything.
* `Lever`: on/off, linked door eids.
* `Trap`: hidden (detect DC), disarm DC, trigger cells, effect (save/damage/condition), one-shot or rearming.
* `Altar`: scenario interactable with hook id and params (healing fountain with regenerating charges,
  donation shrine for public-goods games, relic pedestal, etc.). Implemented in `rules/objects.py`.
* `ExitMarker`: exit square(s); a creature on an exit may `interact` to **escape** (Status.ESCAPED, removed
  from play alive, keeps its loot for terminal payoff).

### 4.2 Creature state

Creature fields (non-exhaustive; the engine agent finalises): `eid, name, template_id, is_pc,
faction (int), controller (ControllerKind), agent_id (str|None), ai (str), size (Size), ctype
(CreatureType), x, y, z, abilities[6] (scores), prof, level, cr, class_levels {class_id: lvl},
subclass, race, hp, max_hp, temp_hp, hit_dice_left, base_ac source, speeds {walk, fly, swim,
climb, burrow}, hover, senses {darkvision, blindsight, tremorsense, truesight}, save_profs,
skill_profs, expertise, resistances / immunities / vulnerabilities (bitmasks over DamageType),
resist_nonmagical_physical, condition_immunities, conditions {Condition: [ConditionInstance]},
exhaustion, status (Status flags), death_successes, death_failures, concentration (effect ref),
effects [ActiveEffect], features [feature ids + per-feature state], resources {id: current},
spell_slots [9], pact_slots (count, level), spells (ids), cantrips, inventory [ItemInstance],
equipped {main, off, armor, shield, ring, cloak, amulet}, gold, light_source (bright, dim),
reaction_available, legendary_left, recharge_ready {action id: bool}, uses_left {action id: n},
stealth_total (if hidden), stats (counters for metrics), grudges {eid: damage dealt to me}`.

Computed properties (cached, invalidated by a per-creature `version` counter): `ac`, `speed`,
`passive_perception`, `spell_save_dc`, `spell_attack_bonus`, `abilities_list`.

### 4.3 Turn flow

* **Exploration mode** (`GameMode.EXPLORATION`): no hostile creatures are aware of each other.
  Creatures take turns round-robin (order: descending DEX score, ties by eid), with full
  movement and action economy (so the same atomic action interface works everywhere). Unaware
  scripted monsters idle/patrol. A "round" still counts (6 s).
* **Combat starts** when any creature becomes aware of a hostile (it sees a non-hidden hostile
  creature, or is attacked/damaged, or hears combat within 12 squares). Everyone rolls
  initiative (d20 + DEX mod + bonuses, ties: higher DEX, then rng). Creatures that were not
  aware of any hostile at the moment combat began are **surprised** (no actions/reactions on
  their first turn). Event COMBAT_START.
* Each creature's **turn**: TURN_START (start-of-turn effects: regeneration, ongoing damage,
  death save if dying, zone effects, condition durations), then the decider issues atomic
  actions until END_TURN; TURN_END (end-of-turn saves, effect expiry). Rounds increment when
  the order wraps (ROUND_START).
* **TurnState**: `movement_left` (feet), `action_used`, `bonus_used`, `free_interaction_used`,
  `attacks_left` (remaining attacks of the Attack action / Multiattack sequence as a list of
  allowed action ids), `cast_leveled_bonus_spell` (bonus-action spell rule), `cast_any_spell`,
  `said` (one SAY per turn), `dashed`, `disengaged`, `action_surge_used`, `sneak_attack_used`,
  `free_toggles` used etc.
* **Combat ends** when no two hostile, aware, conscious, present creatures remain that can see or
  reach each other for 2 consecutive rounds, or when one side is eliminated. Event COMBAT_END,
  back to exploration.
* **Auto end turn**: if the decider has no legal action except END_TURN the engine ends the
  turn automatically. Downed (dying) creatures' turns are processed automatically (death save)
  without asking their controller. Incapacitated creatures likewise.
* **Episode end**: objective resolution (4.15), all agent creatures dead/escaped/surrendered,
  `max_rounds` or `max_steps` reached (truncation).

### 4.4 Engine actions

```python
class ActKind(IntEnum): END_TURN=0, MOVE=1, ASCEND=2, DESCEND=3, ABILITY=4, APPROACH=5, SAY=6, DECIDE=7

@dataclass(slots=True)
class Act:
    kind: ActKind
    arg: int = -1          # MOVE: Direction; ABILITY: index into world.abilities(eid); SAY: token; DECIDE: option
    target: int = -1       # target eid (ABILITY/APPROACH); -1 when point or none
    point: tuple[int, int] | None = None   # target square for POINT abilities (or None)
```

* MOVE: one square in a Direction (5 ft, x2 for difficult terrain, crawling when prone x2,
  swimming/climbing x2 without the speed). Triggers opportunity attacks, zones, traps, hazards.
* ASCEND/DESCEND: change altitude by one square (fliers only; costs 5 ft).
* APPROACH(target): path (Dijkstra, respecting occupancy and costs) toward the target and move as
  far as possible along it this step until adjacent/within the best reach, out of movement, or an
  interrupt (OA, trap, newly spotted enemy, zone entry). Equivalent to a sequence of MOVEs.
* ABILITY(k, target|point): use ability k (4.5).
* SAY(token): free communication token (one per turn; vocabulary size from env config; tokens are
  meaningless to the engine but logged as SAY events and visible to all creatures within 12
  squares or all agents, configurable).
* DECIDE(option): answer a pending decision (4.8).

### 4.5 Abilities

`world.abilities(eid) -> list[Ability]` returns the creature's runtime ability list. The first
`N_UNIVERSAL = 26` entries are the **universal abilities** in this fixed order (their index is the
same for every creature; unavailable ones are simply masked):

| idx | id | cost | target | notes |
|---|---|---|---|---|
| 0 | dash | ACTION | NONE | +speed movement this turn |
| 1 | disengage | ACTION | NONE | no OAs this turn |
| 2 | dodge | ACTION | NONE | attacks vs you have disadvantage, DEX saves advantage (until your next turn) |
| 3 | hide | ACTION | NONE | Stealth vs passive Perception of hostiles that could see you; needs obscurement or no LOS |
| 4 | search | ACTION | NONE | Perception check: reveal hidden creatures / traps / secret doors in sight |
| 5 | help | ACTION | CREATURE | ally: advantage on its next check; enemy within 5 ft: next ally attack vs it has advantage |
| 6 | ready | ACTION | NONE | ready best attack vs first hostile entering reach/range (uses reaction) |
| 7 | stand_up | MOVEMENT | NONE | costs half speed; only when prone |
| 8 | interact | OBJECT_INTERACTION | OBJECT | adjacent door open/close, chest open, loot pickup, lever, altar, exit (escape) |
| 9 | pick_lock | ACTION | OBJECT | thieves' tools: DEX check (+prof if proficient) vs lock DC |
| 10 | force_open | ACTION | OBJECT | STR (Athletics) vs lock DC + 2 (doors/chests) |
| 11 | grapple | ATTACK | CREATURE | Athletics vs Athletics/Acrobatics; target ≤ one size larger |
| 12 | escape_grapple | ACTION | NONE | Athletics/Acrobatics vs grappler's Athletics |
| 13 | shove_prone | ATTACK | CREATURE | contest; target prone |
| 14 | shove_push | ATTACK | CREATURE | contest; push 5 ft away |
| 15 | stabilize | ACTION | DOWNED_ALLY | WIS (Medicine) DC 10, adjacent dying creature |
| 16 | drink_potion | ACTION | SELF | best healing potion in inventory |
| 17 | administer_potion | ACTION | CREATURE | feed a healing potion to adjacent creature (incl. downed) |
| 18 | give_gold | OBJECT_INTERACTION | CREATURE | give `rules.give_gold_chunk` gp (or all if less) to adjacent creature |
| 19 | give_potion | OBJECT_INTERACTION | CREATURE | give one potion to adjacent creature |
| 20 | offer_truce | FREE | CREATURE | propose truce between your faction and target's faction (visible creature) |
| 21 | accept_truce | FREE | CREATURE | accept a pending offer from target's faction -> Relation.TRUCE |
| 22 | parley | ACTION | CREATURE | scripted hostiles: CHA (Persuasion) vs DC from morale -> truce/stand-down |
| 23 | surrender | ACTION | NONE | yield (Status.SURRENDERED) |
| 24 | rest | FREE | NONE | exploration only: toggle short-rest vote (rest happens when all living party members vote) |
| 25 | unarmed_strike | ATTACK | CREATURE | 1 + STR bludgeoning (monk: martial arts die) |

After the universal block come up to `N_SPECIFIC = 24` creature-specific abilities, in this order:
weapon attacks (per carried weapon: melee; thrown variant; ranged; off-hand bonus attack for
two-weapon fighting), monster actions (attacks, save actions, specials; multiattack is expressed
through `attacks_left`), active class/race features (Second Wind, Action Surge, Rage, Cunning Action
variants, Flurry of Blows, Lay on Hands, Channel Divinity options, Bardic Inspiration, Wild Shape,
metamagic toggles, breath weapon...), cantrips, leveled spells (ascending level), item uses (wands,
scrolls, thrown consumables). Overflow beyond 24 is dropped with a logged warning (data should
avoid it). The list must be **stable within a turn** (it may change between turns, e.g. after
picking up a wand); `Creature.version` changes whenever it does.

```python
class Ability:  # runtime, lightweight; shared per definition where possible
    id: str; name: str; kind: str           # "basic"|"weapon"|"spell"|"feature"|"item"|"monster"
    cost: ActionCost; target: TargetType
    range_sq: int; long_range_sq: int; reach_sq: int
    area: AreaSpec | None
    spell_level: int                        # 0 cantrip, -1 not a spell; slot chosen per rules.upcast
    source: object                          # WeaponDef | SpellDef | AttackDef | FeatureDef | ItemDef | ...
    icon: str; vfx: str; tags: tuple[str, ...]
    def usable(self, world, actor) -> bool                        # resources, economy, conditions
    def valid_target(self, world, actor, eid: int) -> bool
    def valid_point(self, world, actor, x: int, y: int) -> bool
    def execute(self, world, actor, target_eid: int, point) -> None
```

Engine helpers used by the env (vectorised, fast):

```python
world.ability_mask(eid) -> np.ndarray[bool]            # len(abilities)
world.target_mask(eid, k, cand_eids: np.ndarray[int], cand_points: np.ndarray[int] (P,2)) -> np.ndarray[bool] (len(cand)+P)
world.move_mask(eid) -> np.ndarray[bool]               # [10]: 8 directions + ascend + descend
world.approach_mask(eid, cand_eids) -> np.ndarray[bool]
world.estimate(eid, k, target_eid) -> (p_hit_or_fail_save, expected_damage)   # analytic, no rng
```

### 4.6 Resolution (core/resolution.py)

* **Attack roll**: d20 (advantage/disadvantage cancel; Halfling Lucky rerolls 1s) + ability mod +
  proficiency (if proficient) + magic bonus + effects (Bless 1d4, Archery +2...). Natural 20 hits &
  crits (Champion 19–20), natural 1 misses. Compare vs target AC + cover bonus.
  Advantage sources: target prone & attacker within 5 ft, target restrained/paralyzed/stunned/
  unconscious/blinded, attacker unseen/hidden/invisible, Help, Reckless Attack, Pack Tactics,
  Faerie Fire, flanking (optional). Disadvantage sources: attacker prone, poisoned, restrained,
  blinded, frightened (source in sight), target dodging, ranged attack with hostile within 5 ft,
  long range, target unseen/invisible, heavy armor stealth n/a, sunlight sensitivity, etc.
  Attacks against paralyzed/unconscious targets from within 5 ft are automatic crits.
* **Damage**: roll dice (crit doubles dice), add modifiers, then per damage component apply
  immunity → resistance (halve, round down) → vulnerability (double). Temp HP absorbs first. Then
  concentration check (DC max(10, dmg/2)). At 0 HP: monsters die (`rules.monsters_die_at_zero`),
  PCs fall unconscious & dying; massive damage (remaining ≥ max HP) = instant death.
* **Saving throw**: d20 + ability mod + prof (if proficient) + effects vs DC. Auto-fail STR/DEX when
  paralyzed/stunned/unconscious. Advantage sources (Magic Resistance, Dwarven Resilience,
  Gnome Cunning, Brave, Danger Sense, Dodge for DEX, cover bonus for DEX saves vs point-origin areas).
* **Ability checks / contests**: skill proficiency, expertise, Jack of All Trades, Reliable Talent n/a (L11).
* **Conditions**: full SRD semantics, durations in rounds (ticked at the bearer's turn end unless
  stated "until the end of the source's next turn"), save-ends, escape DCs. Exhaustion levels 1–6.
* **Modifier hooks**: features/effects/traits/zones implement any of these methods; the engine calls
  only objects registered for that hook (precomputed per creature; cheap):

```
on_turn_start(world, me)          on_turn_end(world, me)
attack_mods(world, ctx)           # ctx: AttackCtx(attacker, target, ability, kind, adv_sources, dis_sources, bonus, crit_min)
defense_mods(world, ctx)          # target-side modifiers to an incoming attack (AC bonus, disadvantage)
damage_mods(world, ctx)           # ctx: DamageCtx(attacker, target, ability, components[], crit, is_spell); add dice, rerolls
incoming_damage_mods(world, ctx)  # target-side: resistances from rage, uncanny dodge halving etc.
save_mods(world, ctx)             # SaveCtx(saver, ability, dc, source, adv, dis, bonus)
check_mods(world, ctx)            # CheckCtx(checker, skill/ability, adv, dis, bonus)
on_hit(world, ctx)                # after a hit is confirmed (sneak attack once/turn, smite decision, stunning strike)
on_damaged(world, me, amount, src)
on_reduced_to_zero(world, me, ctx) -> bool  # return True if prevented (Relentless Endurance, Undead Fortitude)
on_kill(world, me, victim)
on_move(world, me, frm, to)
ac_bonus(world, me) -> int        speed_bonus(world, me) -> int
grant_abilities(world, me) -> list[Ability]   # active features / effects contribute abilities
```

### 4.7 Effects, zones, concentration

* `ActiveEffect(id, source_eid, bearer_eid, remaining, params, concentration_of)` is registry-driven
  (`rules/effects.py`: `EFFECTS[id]` class with the hooks above). Examples: bless, bane, shield_of_faith,
  mage_armor, shield (until start of next turn), hunters_mark, haste, slow, faerie_fire, heroism,
  protection_from_evil_and_good, sanctuary, mirror_image, blur, barkskin, aid, rage, dodge, reckless,
  bardic_inspiration (die), guidance, resistance, true_strike, invisibility, enlarge/reduce,
  hold_person (paralyzed + save ends), etc.
* `Zone(zid, kind, cells: set, center, radius, source_eid, remaining, params, attached_to)` from
  `rules/zones.py`: fog_cloud (heavily obscured), darkness (magical darkness, opaque for sight, darkvision
  can't see through), web (difficult, restrained on entry/start w/ DEX save; flammable), entangle,
  spike_growth (2d4 per 5 ft moved), grease, silence, moonbeam (CON save 2d10 radiant on entry/start),
  spirit_guardians (emanation; WIS save 3d8, half speed for enemies), stinking_cloud, sleet_storm,
  flaming_sphere (movable), cloud of daggers n/a (not SRD), wall_of_fire n/a (4th), call_lightning
  (re-targetable storm), daylight, light (attached light), plus hazard zones (burning oil, alchemist's fire).
  Zones affect vision (opacity/obscurement grids), movement costs, and trigger on enter / turn start.
* Concentration: a creature concentrates on at most one spell; casting another ends the first;
  damage → CON save; incapacitation or death ends it. Ending concentration removes all linked effects
  and zones (EFFECT_END / ZONE_END / CONCENTRATION_END events).

### 4.8 Decisions & reactions (core/decisions.py)

Out-of-turn choices (opportunity attacks, Shield, Uncanny Dodge, Hellish Rebuke, Counterspell,
Protection style, readied triggers) and on-hit riders (Divine Smite, Stunning Strike) are
**decisions**. The resolution code is plain synchronous code that calls:

```python
option = world.decide(decider_eid, DecisionKind, options: list[DecisionOption], context)
```

* If the decider is scripted, or `rules.reaction_mode == "auto"`, the configured auto-decider
  (`world.auto_decider`, default heuristic in `rules/decisions_auto.py`, overridable by AI) answers
  immediately.
* If the decider is an RL agent and `reaction_mode == "agent"`: the engine uses **deterministic
  replay**. Before executing an action that may trigger agent decisions, `World.step` snapshots the
  world (cheap pre-check: an agent creature with a usable reaction within 12 squares of the actor, or
  the actor itself has an on-hit rider). If `decide` needs an answer that is not in the recorded
  answer list, it raises `NeedDecision`; `World.step` restores the snapshot (same RNG state!), stores
  `world.pending_decision = Decision(eid, kind, options, ...)` and returns. The decider then sends
  `Act(DECIDE, option)`; `World.step` restores the snapshot again, appends the answer, and re-executes
  the original action — the dice repeat exactly, so the decider "saw the roll" just as in tabletop play.
  Multiple decisions in one action are answered in order.
* Option 0 is always "decline". Up to 8 options (`MAX_DECISION_OPTIONS = 8`).

### 4.9 Vision, light and stealth

* `GameMap.opaque` (bool [H,W]) = sight-blocking terrain + closed doors + opaque zones (darkness,
  fog for sight purposes). Recomputed incrementally on change (version counter).
* Light grid (LightLevel per cell): ambient level + light sources (map lights, carried torches,
  light-emitting spells/zones/hazards), bright radius then dim radius, blocked by opacity.
* A creature **sees a cell** if it is within its FOV (symmetric shadowcasting, radius
  `rules.sight_radius`) and the cell is not dark for it (bright/dim, or within darkvision range
  (treat dark as dim), or within blindsight/truesight). Magical darkness defeats darkvision.
* A creature **sees a creature** if it sees any occupied cell of it and the target is not hidden
  from it (hidden: Stealth total > observer's passive Perception or failed active Search) and not
  invisible (unless truesight/blindsight). Results cached per step: `world.sees[a, b]` bool matrix.
* Unseen attacker: advantage; unseen target: disadvantage. Attacking reveals a hidden attacker.
* Dim light = lightly obscured: disadvantage on Perception (passive −5).
* Awareness: scripted creatures track `aware_of: set[eid]` and last known positions; agents
  observe only what they see (plus team-shared vision if `env.vision_sharing == "team"`).

### 4.10 Movement

* Speed in feet; each square costs 5 ft (Chebyshev metric; `rules.diagonal_rule = "5-10-5"` variant
  optional), doubled for difficult terrain / crawling / climbing / swimming without speed.
* Creatures may move through allies' squares (not end there), not through hostiles unless 2+ sizes
  apart. Large creatures need clearance (all occupied squares valid).
* Leaving a hostile's reach without Disengage provokes an opportunity attack (decision for the reactor).
* Forced movement (shove, Thunderwave, Repelling Blast) doesn't provoke; falling into pits/lava applies.
* Flying: `z` altitude in squares; ASCEND/DESCEND cost 5 ft; distance between creatures is 3D
  Chebyshev (max of dx, dy, dz edge distances); melee needs reach ≥ distance. A flier that is knocked
  prone or loses its fly speed falls (1d6 per 10 ft) unless it can hover.
* Hazards: lava (`rules.lava_damage` on enter / start of turn), deep water (swim), pits (fall, removed
  unless flying), traps (on enter).

### 4.11 Death and dying (PCs)

0 HP → Condition.UNCONSCIOUS + Status.DYING, drop prone. Start of turn: death save (d20 ≥ 10 success,
1 = two failures, 20 = regain 1 HP). Damage while at 0 HP = 1 failure (crit = 2); 3 successes →
STABLE; 3 failures → DEAD. Healing from 0 → conscious (REVIVED). Stabilize (Medicine DC 10 /
Spare the Dying) → STABLE. Dead creatures drop a LootPile (inventory + gold) if `rules.drop_loot_on_death`.

### 4.12 Items, loot & economy

Gold (gp, integer) and items are personally owned. Pick-up/give/administer actions move them.
Item value (`ItemDef.value_gp`) contributes to personal wealth. Consumables: potions (2d4+2 etc.),
scrolls (cast spell, consumed), wands (charges), thrown flasks (alchemist's fire: 1d4 fire/turn until
extinguished; acid; holy water vs fiends/undead), torches (light), thieves' tools, keys.

### 4.13 Diplomacy & morale

* `world.relations: np.ndarray[int8] [F, F]` (Relation), symmetric. Allied factions share
  "ally" semantics (healing targets, Help). TRUCE: hostile factions that agreed not to attack;
  scripted members won't attack truced creatures.
* **Betrayal**: any attack/harmful effect on an ALLIED or TRUCE creature emits BETRAYAL; if
  `rules.betrayal_breaks_truce` the truce becomes HOSTILE between those factions (and, intra-faction,
  the victim's `grudges` record it). PvP within a faction is allowed when `rules.allow_pvp`.
* Truce offers: `offer_truce` records `(offerer_faction, target_faction, round)`; valid for 2 rounds;
  any member of the target faction may `accept_truce`. Scripted factions decide via their AI (morale,
  relative strength, bribes received via give_gold).
* Morale (scripted): when HP fraction < `MonsterDef.morale` or the leader dies or half the group is
  down, WIS save DC 10; failure → FLEE (move away, Dash/Disengage) or SURRENDER if `can_surrender`.
  Attacking a surrendered creature is legal (reward components let researchers price "mercy").
* `parley`: CHA (Persuasion/Intimidation, best) check vs DC 10 + 2·(their strength ratio) − bribes;
  success → TRUCE with that faction.

### 4.14 Rest

Exploration only. When every living, conscious, present member of a faction (agents and scripted)
has voted `rest` and no hostile is aware of them, a **short rest** occurs: `rules.short_rest_rounds`
pass (zones/effects with shorter duration expire), each member spends hit dice (heuristic: while
HP < 50% and dice left), short-rest resources recharge (Second Wind, Action Surge, Ki, Channel
Divinity, warlock slots, Wild Shape, Bardic Inspiration at L5, Arcane Recovery once). Optional
wandering-monster check (`rules.wandering_monster_chance`). Votes reset after the rest or when combat starts.

### 4.15 Objectives & episode end

`ObjectiveSpec`s are evaluated after every atomic action (cheap counters). Completion emits
OBJECTIVE events; an objective with `ends_episode` terminates the episode. Outcome:
`world.outcome = {"reason", "winners": [faction idx], "objectives": [...]}`.

### 4.16 Cloning, determinism, performance

* `world.clone()`: deep copy of mutable state; shares definitions and static map arrays
  (`GameMap.static`). Must be < 2 ms for a 40×40 map with 12 creatures.
* `world.snapshot()/restore(snapshot)` used by decisions; `restore` keeps the World object identity.
* Caches (vision, light, sees-matrix, ability lists, masks) are keyed by version counters and are
  part of the state (or rebuilt lazily after cloning).
* Hot paths: FOV/LOS/cover/dijkstra in `core/geometry.py` & `core/vision.py` (numba with
  `cache=True`, pure-NumPy fallback when numba is missing).

### 4.17 Public engine API (used by env, AI, scenarios, tests)

```python
from emberdeep.core.world import World, RulesConfig
from emberdeep.core.actions import Act, ActKind           # (may live in world.py and be re-exported)

world = World(map_data: MapData, rules: RulesConfig, seed: int)   # usually via scenarios.builder.build_world(spec, ...)
eid = world.spawn_creature(creature)                       # factory builds Creature from MonsterDef / CharacterBuild
world.add_object(obj) -> eid
world.set_relation(fa, fb, Relation)
world.start()                                              # EPISODE_START; combat if start_mode == "combat"

world.decider() -> int        # eid that must act now (pending decision holder, else active creature), -1 if done
world.active -> int           # creature whose turn it is
world.pending_decision -> Decision | None
world.step(act: Act) -> None  # apply for the decider; advances turns/rounds; auto-handles dying/incapacitated turns
world.drain_events() -> list[Event]
world.done -> bool; world.truncated -> bool; world.outcome -> dict
world.round, world.mode, world.step_count

world.abilities(eid) -> list[Ability]
world.ability_mask(eid); world.target_mask(eid, k, cand_eids, cand_points); world.move_mask(eid); world.approach_mask(eid, cand)
world.estimate(eid, k, target_eid) -> tuple[float, float]
world.creature(eid) -> Creature; world.creatures_present() -> list[Creature]; world.objects_present() -> list[WorldObject]
world.sees(a, b) -> bool; world.visible_cells(eid) -> np.ndarray[bool]; world.light -> np.ndarray[uint8]
world.distance(a, b) -> int (squares); world.cover(a, b) -> Cover; world.relation(a, b) -> Relation
world.clone() -> World; world.snapshot(); world.restore(snap)
world.auto_decider: Callable[[World, Decision], int]
```

---------------------------------------------------------------------------------------------

## 5. Rules registries (emberdeep/rules)

Each registry maps string ids (from data hooks) to implementation classes/functions:

* `rules/features.py`  `FEATURES[id]` — class features (Rage, Sneak Attack, Divine Smite, Extra Attack,
  Fighting Styles, Martial Arts, Ki options, Wild Shape, Bardic Inspiration, Cutting Words, metamagic, invocations...).
* `rules/traits.py`    `TRAITS[id]` — racial & monster traits (pack_tactics, nimble_escape, sunlight_sensitivity,
  regeneration, undead_fortitude, magic_resistance, keen_senses, aggressive, relentless_endurance, lucky,
  brave, fey_ancestry, dwarven_resilience, gnome_cunning, hellish_resistance, savage_attacks, breath_weapon...).
* `rules/effects.py`   `EFFECTS[id]`
* `rules/zones.py`     `ZONES[id]`
* `rules/spells.py`    `SPELL_HOOKS[id]` — bespoke spell logic (misty_step, counterspell, dispel_magic, haste,
  revivify, spiritual_weapon, flaming_sphere, call_lightning, conjure_animals, animate_dead, shield, sleep,
  color_spray (HP pools), magic_missile, scorching_ray, eldritch_blast beams, thunderwave push...). Spells
  without a hook are executed by the **generic spell interpreter** from their SpellDef fields
  (attack/save/damage/healing/conditions/effect/zone/area/targets/upcast).
* `rules/items.py`, `rules/objects.py` — item uses and altar/trap/lever behaviours.

Every id referenced anywhere in `data/` must exist in the matching registry; `tests/test_registry.py`
enforces it.

---------------------------------------------------------------------------------------------

## 6. Geometry / vision / pathing API (core/geometry.py, core/vision.py, core/pathing.py)

All functions take NumPy arrays, are pure, and are numba-jitted when numba is available.

```python
# geometry.py
chebyshev(ax, ay, bx, by) -> int
creature_distance(ax, ay, asz, az, bx, by, bsz, bz, rule=0) -> int    # edge-to-edge squares (adjacent = 1), 3D for z
line_cells(x0, y0, x1, y1) -> np.ndarray (n, 2)                       # Bresenham, inclusive
has_los(opaque, x0, y0, x1, y1) -> bool                               # centre-to-centre, permissive & symmetric
cover_between(opaque, half_cover, tq_cover, ax, ay, asz, bx, by, bsz, creature_occ) -> int   # Cover; DMG corner method
area_cells(shape, ox, oy, tx, ty, size_sq, width_sq, opaque, H, W) -> np.ndarray (n, 2)       # SRD templates; cells whose
                                                                       # centre lies inside; blocked by total cover from origin
# vision.py
fov(opaque, x, y, radius) -> np.ndarray[bool] (H, W)                  # symmetric shadowcasting
fov_multi(opaque, cells (n,2), radius) -> bool (H, W)                 # union for large creatures
light_grid(opaque, ambient: int, lights: np.ndarray (n, 4) [x, y, bright_sq, dim_sq], darkness_mask) -> uint8 (H, W)
# pathing.py
dijkstra(cost: float32 (H,W) [inf = blocked], sx, sy, max_cost) -> float32 (H,W)
path_from(dist, sx, sy, tx, ty) -> np.ndarray (n, 2)                  # reconstruct by descent
inflate_blocked(blocked: bool (H,W), size) -> bool (H,W)              # clearance for size x size creatures
```

---------------------------------------------------------------------------------------------

## 7. RL environment API (emberdeep/env)

### 7.1 Entry points

```python
import emberdeep
env = emberdeep.make_aec("goblin_ambush", render_mode=None, **config)       # PettingZoo AECEnv
env = emberdeep.make_parallel("dragon_hoard", **config)                    # PettingZoo ParallelEnv
env = emberdeep.make_gym("goblin_ambush", agent="party_0", others="scripted") # Gymnasium single-agent
venv = emberdeep.make_vector("goblin_ambush", num_envs=8, api="parallel")   # subprocess vector env
emberdeep.list_scenarios()
```

`EnvConfig` (env/config.py) fields: scenario, render_mode (None | "human" | "rgb_array"), seed,
max_entities (16), n_point_targets (16, = 8 directions x {3, 6} squares), view_radius (10 → 21x21),
comm_vocab (8, 0 = off), reaction_mode ("auto" | "agent"), vision_sharing ("none" | "team"),
enemy_hp ("bucket" | "exact" | "hidden"), social_features (True), entity_order ("distance" | "fixed"),
upcast ("lowest" | "highest"), reward overrides (weights, prosocial/svo, team_reward, terminal payoff),
skip_scripted (True: scripted turns resolved inside step), include_state_in_obs (False), obs_dtype
(float32), flatten_obs (False), rules overrides, render options (width, height, fps, speed, sound, pov).

### 7.2 Agents and turn semantics

Agents are the scenario's agent-controlled creatures (`agent_id`, e.g. "party_0"). D&D is turn
based, so:

* **AEC**: `agent_selection` is the current decider (active creature or pending decision holder).
  The agent keeps being selected while its turn lasts (several atomic actions). Scripted creatures'
  turns are executed inside `step()` between agent decisions. Dead/escaped agents are terminated;
  downed agents are not (they may be revived) but are skipped until conscious.
* **Parallel**: every step all live agents submit an action; only the decider's action is applied,
  every other agent's mask contains only index 0 (NOOP) and `infos[a]["acting"] = False`. Rewards are
  delivered to all agents every step (credit for events that happened since their last step).

### 7.3 Action space (flat Discrete with masks)

`K = 50` ability slots (26 universal + 24 specific), `T = max_entities + n_point_targets = 32` target
slots, `V = comm_vocab`, `D = 8` decision options.

```
0                          END_TURN (NOOP when you are not the decider)
1..8                       MOVE N, NE, E, SE, S, SW, W, NW
9, 10                      ASCEND, DESCEND
11 + k*T + t               ABILITY k on target slot t   (k < K, t < T)
11 + K*T + e               APPROACH entity slot e        (e < max_entities)
11 + K*T + E + v           SAY token v
11 + K*T + E + V + d       DECIDE option d
```

Target slots `t < max_entities` are the entity slots of the observation (slot 0 = self);
`t >= max_entities` are point targets (relative squares). Entity-centred POINT abilities aim at the
entity's square. The observation also exposes the factored masks `ability_mask [K]` and
`target_mask [K, T]` so autoregressive policies can use them.

### 7.4 Observation (Dict space, per agent)

* `self` [F_self] float: HP fraction, temp HP, AC, speed, movement left, action/bonus/reaction flags,
  attacks left, position (normalised), altitude, conditions (14), status bits (18), exhaustion,
  concentration, death saves, spell slots (9) + pact slots, class resources (normalised), gold
  (log-scaled), potions, class one-hot (12) / monster flag, level, ability mods (6), mode
  (exploration/combat), round, is-decider, decision kind one-hot, rest votes, faction one-hot (≤ 8).
* `entities` [max_entities, F_ent] float + `entity_mask` [max_entities]: per visible entity (slot 0 =
  self): present, kind one-hot, relation one-hot (self/ally/neutral/truce/hostile), relative dx, dy,
  dz, distance, in-reach, in-range-of-my-best-attack, has-LOS, cover level, HP (by `enemy_hp`), AC (allies
  only unless revealed), conditions (14), statuses, size, creature type one-hot, CR/level, is agent,
  is active, initiative position, concentrating, downed, surrendered, fleeing, truce offer pending,
  last SAY token one-hot, **social history** (damage it dealt to me / my allies, healing it gave me /
  my allies, gold it gave me, betrayals by it — normalised), object state (door/chest/loot value).
* `entity_ids` [max_entities] int: template vocabulary ids (for embeddings).
* `abilities` [K, F_ab] float: available, kind, cost, target type, range, area, avg damage, damage
  type, healing, spell level, uses left, concentration, save ability, attack bonus / DC, tags.
* `ability_ids` [K] int: ability vocabulary ids.
* `map` [C, 2R+1, 2R+1] egocentric (float32 or uint8): in-bounds, wall/opaque, door closed, door open,
  difficult, hazard, water, pit, cover objects, currently visible, explored, light level, allies,
  enemies, neutrals, objects/loot, harmful zone, beneficial zone, reachable-this-turn, threat
  (cells in hostile reach), exit/objective region.
* `messages` [n_agents, V] last SAY token per agent (one-hot, with age decay) if comm enabled.
* `action_mask` [A] int8, `ability_mask` [K], `target_mask` [K, T].
* optional `state` (global state for CTDE) when `include_state_in_obs`.

`env.state()` returns the global state vector (all creatures in fixed order + full map summary +
scenario progress) for centralised critics; `env.state_space` describes it.

### 7.5 Rewards (env/rewards.py)

Per-agent reward components are computed from events each step and exposed in
`infos[a]["reward_components"]`:

`damage_dealt, damage_taken, friendly_damage, kills, kill_xp, team_xp, healing_given, healing_received,
revives, downed, death, gold, item_value, gold_given, gold_received, objective, victory, defeat,
escaped, survived, time, exploration, betrayal, truce, spared, damage_neutral`.

Reward = Σ weights·components (defaults in `rewards.DEFAULT_WEIGHTS`, scenario overrides, env
overrides), then **social mixing**: prosocial α (`r'_i = (1-α) r_i + α · mean_{j≠i, same team} r_j`), or
SVO angle θ (`r'_i = cos θ · r_i + sin θ · mean_{j≠i} r_j`), or an arbitrary mixing matrix W
(`r' = W r`), or fully cooperative `team_reward`. Terminal payoffs (e.g. personal gold if escaped).

### 7.6 Metrics (env/metrics.py)

Episode-level `infos[a]["episode"]` at termination (Melting-Pot style): per-agent return, team
return, collective return (efficiency), equality (1 − Gini of returns), wealth Gini, survival rate,
win/loss, rounds, heals/revives given, gold given, betrayals, truces formed/broken, friendly-fire,
free-riding index (damage-taken share vs loot share), common-pool depletion time (scenario-specific).

### 7.7 Other features

* `env.clone_state()/restore_state()` (wrap `world.clone()`), `env.action_meanings(agent)` (list of
  `ActionDescriptor` for the human UI / debugging), `env.render()` (render adapter → renderer),
  `env.unwrapped.world` access, `emberdeep.replay.Recorder` wrapper (writes `.edr` replays: gzip-pickled
  StaticScene + per-step FrameState + events), `emberdeep.viewer` (replay player), `emberdeep.play`
  (human vs/with AI), `emberdeep.bench` (throughput).

---------------------------------------------------------------------------------------------

## 8. Scripted AI (emberdeep/ai)

`Policy.act(world, eid) -> Act` and `Policy.decide(world, decision) -> int`. A shared **utility AI**
scores candidate (ability, target) pairs with `world.estimate` (expected damage, kill chance, healing
value, control value) minus risk (OA, hazards, ally in AoE), plus movement planning (approach, keep
range, seek cover, flank, retreat to healer). Profiles: brute, skirmisher (hit & run, Nimble Escape,
hide), archer (keep range & cover), caster (AoE clusters avoiding allies, control, concentration
management), support (heal downed allies first, buff), beast (pack tactics, knock prone), leader
(buffs, commands), coward (flees early), ambusher (hides, waits), boss (legendary/recharge timing),
summon. Morale, surrender, truce acceptance, and parley responses live here. **Hero bots** (one per
class) provide baselines and teammates for the single-agent Gym wrapper.

---------------------------------------------------------------------------------------------

## 9. Scenarios & maps (emberdeep/scenarios)

* `ascii_map.parse(MapSpec) -> MapData`; `mapgen.generate(generator, params, seed) -> MapData`
  (BSP dungeon, cave (cellular automata), crypt, arena, forest clearing, sewer, temple); `decorate.py`
  places themed decor and lights.
* `builder.build_world(spec: ScenarioSpec, rules_overrides, seed) -> World` (spawns, factions, relations,
  objects, objectives, agents). PCs come from `data/archetypes.py` builds via `core/factory.py`.
* Library (each with documented game-theoretic structure): goblin_ambush (cooperation baseline),
  dragon_hoard (stag hunt), cursed_coffer (prisoner's dilemma over loot), rival_delvers (two parties,
  relic race: truce vs. fight), fountain_of_life (common-pool resource: regenerating healing charges
  under waves), shrine_of_wards (threshold public good: donate gold for a party-wide ward), hold_the_bridge
  (volunteer's dilemma), gladiator_pit (free-for-all with alliances), caravan_escort (protect NPC, bribe
  bandits), deep_delve (procedural multi-room dungeon crawl: exploration, traps, doors, rests).

---------------------------------------------------------------------------------------------

## 10. Renderer & UI (emberdeep/render)

**Art direction — "illuminated tabletop".** A top-down battle map in the tradition of virtual
tabletops, elevated to video-game polish: moody darkness carved by warm, flickering torchlight;
painterly, normal-mapped procedural stone, wood, water and lava; circular creature tokens with
beveled metal rims, team-coloured inner rings and ivory silhouettes (game-icons.net), soft drop
shadows; gilded slate UI panels with Cinzel headings and Alegreya Sans body text.

**Tech.** pygame-ce for window/input/fonts/surfaces, moderngl (OpenGL 3.3 core; macOS forward-compat
core profile) for the world: G-buffer-lite (albedo, normal, emissive) → per-light additive pass with
ray-marched soft shadows over an occlusion texture and normal-mapped N·L → fog-of-war composite
(unexplored = black with drifting mist, explored = desaturated memory, visible = full) → tokens & props
→ particles (instanced, additive/alpha) → zone overlays (animated shaders) → bloom (dual-filter),
ACES tonemapping, per-theme colour grading, vignette, film grain, screen shake / hit-stop → HUD
surface overlay. `rgb_array` mode renders into an offscreen framebuffer (standalone context).
Quality presets (low/medium/high) scale light map resolution, shadow ray steps, particle counts, bloom.

**Director.** Events become timed animation tracks (move along path with easing & dust; melee
lunge + swoosh + impact flash, sparks, knockback; projectiles with trails; per-spell VFX recipes keyed
by `vfx`; dynamic flash lights; damage numbers; condition pop-ups; deaths (token shatter/fade, soul
wisp, corpse decal); door/chest animations; coin arcs for gold transfers; speech bubbles). Playback
speed 0.25×–16×, pause/step, skip animations; camera follows the active creature with a critically
damped spring, zoom/pan, auto-framing in combat, cinematic punch-in on crits/kills (optional).

**HUD (render/ui).** Initiative ribbon with portraits; party frames (HP/temp HP, AC, conditions,
concentration, slots/resources, live RL return); combat log with dice breakdowns and damage-type
colours; hover tooltips (stat cards); selection inspector (full sheet incl. inventory and reward
components); minimap; POV selector (omniscient / each agent / each faction); overlays (grid,
threat ranges, LOS, light); research overlay (value estimate & top-k action probabilities from
`FrameState.annotations`); episode-end scoreboard with social metrics; human play action bar with
hotkeys, valid-target highlighting, AoE previews, path preview, End Turn. Procedural audio SFX
(numpy-synthesised) and ambient loop, mutable.

---------------------------------------------------------------------------------------------

## 11. Testing & performance targets

* Unit tests for rules (attack math, advantage stacking, crits, resistances, death saves,
  concentration, OAs, conditions, each spell family, each class feature), registry completeness,
  determinism (same seed ⇒ same trajectory hash), clone independence, PettingZoo `api_test` /
  `parallel_api_test`, Gymnasium `check_env`, obs-space containment, mask correctness (every
  masked-in action is executable without error), random-agent fuzzing (10k episodes no crash).
* Benchmarks (`emberdeep.bench`): steps/s per core for random & scripted policies; target ≥ 2k/s
  on goblin_ambush (4 PCs vs 6 goblins), ≥ 20k/s with 8-process vector env on a laptop.
* Rendering: 60 FPS at 1600×900 on integrated GPUs with "medium" preset.

## 12. Licensing

Code MIT. Rules content: SRD 5.1 by Wizards of the Coast, CC-BY-4.0 (attribution in NOTICE.md).
Icons: game-icons.net (CC BY 3.0, authors in assets/icons/_index.json). Fonts: SIL OFL. All maps,
scenarios, textures, sounds and VFX are original/procedural. Emberdeep is not affiliated with or
endorsed by Wizards of the Coast; it uses no trademarks or non-SRD content.
