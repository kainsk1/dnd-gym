# Emberdeep — rules interpretations and deviations (engine)

Emberdeep implements SRD 5.1 (CC-BY-4.0) for character levels 1–5. Where tabletop play needs a judgement call or the
atomic-action interface forces a simplification, the choice made by the engine is listed here. Section numbers refer to
`DESIGN.md`.

## Dice and randomness
* Every die is `int(rng.random() * sides) + 1` on `world.rng` (a PCG64 `numpy.random.Generator`); no other RNG is used.
  A crit doubles the dice, never the flat modifier. Advantage/disadvantage roll two d20s at once (Halfling *Lucky*
  rerolls each natural 1 before the high/low pick).
* Unarmed strike: `1 + STR mod` (minimum 0) bludgeoning, proficient. Martial Arts dice come from the `unarmed_damage` hook.

## Turn flow (§4.3)
* **Exploration rounds.** Creatures take full turns round-robin (descending DEX, ties by eid); a round counts as 6 s of
  game time and counts towards `max_rounds`. `rules.exploration_turns=False` ends an exploration turn automatically after
  the creature used its action.
* **Combat start.** Detected when a hostile pair sees each other, when an attack/harmful effect lands, or when combat is
  heard within `hearing_radius`. If it starts in the middle of someone's exploration turn, that creature finishes its
  turn, then round 1 starts at the top of the initiative order. Initiative = d20 + DEX mod (+ `initiative_bonus` hooks);
  ties: higher DEX score, then a seeded random draw.
* **Surprise.** Anyone not aware of any hostile when initiative is rolled is surprised: its first turn is skipped
  (no movement, action, bonus action or reaction until that turn ends). `rules.surprise=False` disables it.
* **Combat end.** Two consecutive round boundaries with no hostile, conscious, non-surrendered pair that is aware of /
  can see each other, or immediately when one side is gone. Dying, stable and surrendered creatures do not keep combat alive.
* **Auto end turn.** The engine ends a turn by itself when the creature is dying/stable/surprised/incapacitated, or has
  no movement and no usable action other than FREE abilities (give/truce/rest) and END_TURN. Creatures that are merely
  *able* to use a free object interaction keep their turn (the agent ends it).
* **Dying creatures** roll their death save at the start of their turn automatically; stable creatures simply pass.
  A stable creature regains 1 HP after 1d4 hours (see v1.1 additions).
* **Round counters.** Effect/zone durations count down at `ROUND_START`; conditions at the *end of the bearer's turn*;
  "until the end of X's next turn" is tracked per condition/effect with an *armed* flag so the turn in which it was
  applied does not count.

## Actions and economy (§4.4–4.5)
* **Free weapon switching.** Every carried weapon is usable at any time (one ability per weapon, plus a thrown variant
  and an off-hand bonus attack); drawing/sheathing costs nothing. Versatile weapons use the two-handed die when no
  shield is equipped. Thrown weapons leave the inventory and land as a `LootPile` under the target.
* **Ammunition.** A creature that carries *any* `ammo` item consumes one per shot and cannot fire without it; creatures
  without any ammo item have an unlimited quiver. Loading weapons end the Attack action after one shot.
* **Attack action.** The first attack-cost ability commits the Action; `attacks_left` (Extra Attack, from
  `attacks_per_action` hooks) or the remaining Multiattack sequences (monsters) gate the rest. Movement may happen between
  attacks. Grapple, shove and unarmed strike each replace one attack. Haste's extra action allows one weapon attack, Dash,
  Disengage, Hide, Interact.
* **Bonus-action spell rule** is symmetric: after casting a bonus-action spell only 1-action cantrips may follow; a
  leveled action spell cast earlier in the turn forbids a bonus-action spell.
* **Object interaction.** One free per turn; later ones spend the Action. Opening a door never needs a check; locked
  doors/chests need a key (interact), Thieves' Tools (DEX check, +proficiency if proficient) or force (Athletics vs lock DC + 2).
* **Help** is two effects: on an ally, advantage on its next ability check *or* attack (lasts until the helper's next
  turn); on an adjacent enemy, the next attack by an ally of the helper has advantage.
* **Ready** stores the creature's best melee and best ranged attack; the first hostile that comes into reach/range
  (movement, forced movement or teleport) triggers a `READIED_TRIGGER` decision for the readier. The readied action
  and its reaction vanish at the start of the readier's next turn.
* **Hide.** Needs, against every hostile that could see you, total concealment: no line of sight, heavy obscurement /
  darkness at your square (or the observer is blinded), ≥ 3/4 cover, or a `can_hide` trait hook (e.g. Naturally Stealthy).
  Roll Stealth (disadvantage in noisy armor) against each such observer's passive Perception. A hidden creature is
  unseen by observers whose passive Perception is below its Stealth total unless they *found* it (Search succeeded);
  it is automatically seen by anyone with a clear view (`hide_possible` false). Attacking or casting reveals it.
* **Search** is a Perception check against hidden creatures in line of sight (stealth total) and hidden traps (DC).
  Passive Perception also notices traps within 2 squares while walking. Dim light or lightly obscuring terrain gives
  disadvantage on Perception (passive −5) for observers without darkvision.
* **Parley.** Only against scripted hostile creatures with INT ≥ 4 within 6 squares. DC = 10 + 2·(their faction power ÷
  yours) − 2, −2 if their average HP is below half, −2 if the target is below its morale threshold, − gold your faction
  has given theirs ÷ 10 (max −10); clamped to 5–25. Best of Persuasion/Intimidation. Success → `TRUCE` between the factions.
  "Power" is the sum of (max(CR, level, ¼) + ½) over their conscious members.
* **Truce offers** last `truce_offer_rounds` rounds. Scripted factions answer through their AI (`accept_truce` ability).
* **Surrender.** A surrendered creature cannot use attack abilities until it is attacked (which clears the flag).
  Attacking a surrendered creature is legal; no engine penalty (reward shaping prices mercy).

## Resolution (§4.6)
* **Advantage sources** implemented in the core: target prone & within 5 ft, target blinded/restrained/paralyzed/
  stunned/unconscious/petrified, attacker unseen by the target (invisible, hidden, darkness, heavy obscurement),
  Help, optional flanking (`rules.flanking`, opposite side, Medium-or-smaller, melee), plus hook-supplied (Pack
  Tactics, Reckless, Faerie Fire …). **Disadvantage:** attacker prone/poisoned/restrained/blinded/frightened (source in
  sight)/exhaustion ≥ 3, target dodging, ranged attack with a hostile able creature within 5 ft, beyond normal range,
  target unseen, heavy weapon used by a Small creature.
* **Auto-crit** against paralyzed/unconscious targets within 5 ft. Natural 20 always hits, natural 1 always misses;
  `crit_min` (Champion) only widens crits.
* **Shield / Uncanny Dodge** are offered *after* the attack roll is known (the decider sees the roll — decisions are
  replayed with identical dice); the ATTACK event is emitted after reactions with the final result.
* **Resistances** are applied per damage component: immunity → resistance → vulnerability, rounding down; temp HP
  absorbs next; petrified creatures resist everything and are immune to poison. Nonmagical B/P/S resistance
  (`resist_nonmagical_physical`) is bypassed by magical or silvered attacks.
* **Instant death** when the damage left after reaching 0 HP is ≥ max HP, also at 0 HP. Damage at 0 HP adds one death-save
  failure (two on a crit). Monsters die at 0 HP (`monsters_die_at_zero`), PCs and NPC-like creatures go dying. Dying and
  stable creatures stay on the map (`present`); dead ones are removed and drop their gold/items as a `LootPile`.
* **Concentration** DC = max(10, damage/2) per damage packet; incapacitation/death ends it; every linked effect,
  zone and condition (`link` = concentration uid) disappears with it.
* **Exhaustion** 1–6 per SRD; level 4 halves max HP (restored when it drops below 4).
* **Frightened** forbids willingly moving closer to a visible source; **charmed** forbids attacking the charmer.

## Vision, light, cover (§4.9)
* Visibility is symmetric line-of-sight (`has_los`) + light + senses. A creature in a heavily obscured square (fog,
  magical darkness, `zone_heavy`) is *effectively blinded* and cannot be seen except by blindsight/truesight/tremorsense
  in range. Magical darkness defeats darkvision; plain darkness is dim for darkvision creatures within range.
* Cover uses the DMG corner method (`geometry.cover_between`); creatures in between give half cover. DEX saves vs area
  effects add the cover bonus computed from the effect's origin point.
* `world.sees` is cached per *vision epoch* (bumped by movement, doors, zones, conditions, light changes).

## Movement (§4.10)
* Chebyshev metric by default; `rules.diagonal_rule = "5-10-5"` alternates 5/10 ft diagonals per turn. Difficult
  terrain, crawling, swimming without a swim speed, zone slowing and dragging a grappled creature each add +1 foot per foot
  (additive, as in the SRD). Climb speeds are modelled for low walls and burrow speeds underground (v1.1 additions).
* A single MOVE never ends in an occupied square. `APPROACH` may *pass through* allied squares (and through hostile
  creatures two sizes apart) but never stops on one; it pathfinds with `pathing.dijkstra`, stops adjacent / at best
  reach, when movement runs out or on an interrupt (opportunity attack, trap, lava, newly started combat).
* Diagonal steps are illegal between two blocked orthogonal neighbours and through door squares.
* Opportunity attacks: any hostile, able, non-surprised creature with a reaction that sees the mover and has a melee
  attack for its reach; none if Disengaged or `rules.opportunity_attacks` is off. Forced movement and teleportation never provoke.
* **Pits** (`Terrain.PIT`) are chasms: forced movement into one deals `rules.pit_damage` fall damage and removes the
  creature from play (not dead). **Lava** deals `rules.lava_damage` when entered and at turn start. Fliers falling
  (prone/incapacitated/fly speed lost) take 1d6 per 10 ft and land prone.

## Spells (§4.5, §5)
* Spells are interpreted from `SpellDef`; slot choice follows `rules.upcast` ("lowest" or "highest" available slot ≥ spell
  level; a warlock pact slot is preferred on ties). Cantrip damage scales with the caster's character level
  (monsters: `caster_level`).
* **Multi-target spells** pick extra targets automatically: helpful spells prefer the most wounded allies in range, harmful
  ones the nearest enemies. **Rays/darts** (Magic Missile, Scorching Ray, Eldritch Blast) all go to the chosen target and
  spill over to the nearest enemy if it drops.
* **Zones deliver their own effects.** If a spell has a `zone`, its save/damage/conditions are applied on enter / at the
  start of turn by the zone class, not at cast time — except `zone.params['on_cast']` and the ids `web`, `grease`,
  `entangle` (creatures inside the area save once when it appears). Creatures already inside a damaging zone when it is
  created are not hit until they enter again or start their turn there.
* **Area templates** come from `geometry.area_cells`: sphere = radius-N disk, cube placed adjacent to the caster (or
  centred on the point when `origin="point"`), cones/lines from the caster's square edge. Friendly fire follows
  `rules.friendly_fire`; area spells only emit BETRAYAL for `TRUCE` relations (never for allies), single-target harm to
  ALLIED/TRUCE creatures always does.
* **Counterspell** is offered to every non-allied creature within 60 ft that sees the caster and knows it (one option
  per usable slot). Shield also protects against Magic Missile (`BLOCKS_MM`).
* Costly material components (`material_cost_gp`) are paid from carried gold; Silence blocks any spell with a verbal component.
* Scrolls are consumed on use; wands lose a charge (and are not consumed unless `consumable`).

## Decisions (§4.8)
* In `reaction_mode="agent"` the engine snapshots before an act that *could* prompt an agent (an agent that can react
  — hook options or reaction spells — is within range, an agent actor has riders, …). When `decide()` needs an answer,
  the world is restored (RNG state included), `pending_decision` is set and the same act is re-executed after
  `Act(DECIDE, i)` with the answer log extended. A decision that arises when no snapshot was taken (pre-check said no)
  is answered by `world.auto_decider` instead of failing.
* Option 0 is always "decline"; at most `MAX_DECISION_OPTIONS = 8` options (one `decline` + 7). Legendary actions
  (`DecisionKind.LEGENDARY`, appended to the enum) are offered after every other creature's turn in combat.

## Diplomacy, morale, rest (§4.13–4.14)
* `world.morale_check(c)` implements "WIS save DC 10 when morale breaks" for scripted AI; failure → FLEE, or SURRENDER for
  `can_surrender` creatures. The engine never triggers it by itself.
* Short rest: when every living, conscious member of a faction has voted `rest` in exploration and none is aware of a
  hostile. Hit dice are spent while HP < 50 %; short-rest resources and pact slots recover; `rules.short_rest_rounds`
  elapse for effect/zone/condition durations *without* advancing `world.round` (so rests do not eat the round budget).
  `world.long_rest()` is available to scenarios.
* A wandering monster may appear after a rest (`wandering_monster_chance`, `wandering_monster_ids`).

## Not modelled
Vehicles, downtime, language barriers (out of scope), suffocation / holding breath, vertical climbing of cliffs/walls (maps are
top-down: only low walls are climbable obstacles), high jumps (only long jumps over pits are automatic), tool proficiencies, and
any non-SRD content.

## v1.1b movement additions
* **Squeezing.** A creature may enter a footprint that overlaps obstacles when its top-left (size-1)^2 squares are free (a Large
  creature in a 1-wide corridor); it pays +1 foot per foot, has disadvantage on attack rolls and DEX saves and attackers have
  advantage (`World.is_squeezing`). Amorphous oozes, air forms and swarms are exempt. The approach planner doubles the cost of
  squeezed anchors.
* **Incorporeal movement** (trait `incorporeal_movement`): walls, pillars, statues and closed doors are difficult terrain
  (10 ft per square); pits, void and map edges still stop it. Ending the turn inside costs 1d10 force.
* **Jumping.** A step into a pit square becomes a long jump when the creature has the distance (STR score ft with a 10-ft straight
  run-up this turn, half standing; the Jump spell triples it) and the movement (5 ft per square travelled) and the landing square
  is free in the same direction. The approach planner only plans jumps a creature can make from a standing start within its
  remaining movement, so masked-in approaches always execute.
* **Per-observer illusions.** `World.discern_illusion(zone, eid)`: Minor Illusion stays opaque for everybody except the creatures
  that discerned it (`World.ext['illusion_seen']`).
* **Sight radius.** Targets farther than `rules.sight_radius` (24 squares = 120 ft) cannot be seen, hence not targeted, even if
  lit; long-range spells (Eldritch Spear 300 ft) need `sight_radius` raised in the scenario rules.

## Pass 2 additions (engine hooks and corrections)
* **Movement/pathing.** `_step_cost`, `move_mask` and the approach planner share one rule set (only "both orthogonal
  neighbours blocked" forbids a diagonal; frightened creatures cannot step closer to the source), so every masked-in
  APPROACH moves at least one square. Hostile creatures are impassable in approach costs; big movers inflate them.
* **Ability list.** Duplicate ability ids are dropped (first wins) and, if more than 24 specific abilities remain,
  the least valuable are cut (`abilities.dedupe_and_prioritise`). Multiattack members cost ATTACK.
* **Concentration** has a duration (`Concentration.remaining`, ticked every round, lapses linked effects/zones).
* **Zero HP.** Hooks with `BEFORE_MASSIVE = True` (Wild Shape) run before the massive-damage test; the overflow is theirs
  to reduce through `ctx.extra["overflow"]`.
* **Blink/etherealness.** Effects flagged `RETURN_AT_TURN_START` return the creature at the start of its own turn.
* **Hooks added** (see `rules/registry.ALL_HOOKS`): `on_spawn`, `force_save_mods`, `after_d20`/`observe_d20`,
  `ally_save_mods`, `modify_healing`, `death_save_mods`, `veto_exhaustion`, `avoids_opportunity`, `can_pass_through`,
  `may_share_space`, `ignores_difficult`, `weapon_ability`, `ability_blocked`, `target_filter`, `economy_mods`,
  `spell_range_mult`, `ignore_components`, `before_cast`, `on_fall`; `projectile_block` is consulted for
  ordinary ranged weapon attacks (see v1.1).
* **Decisions** `SPELL_CHOICE` (spell hooks with `HAS_OPTIONS`) and `ON_FALL` (feather fall) were appended.
* **World API**: `World.summon`, `add_to_order`, `round_hooks`, `ext`, `seal_exit`, `sync_items`; `CharacterBuild` is kept
  on the creature (`Creature.build`).

## v1.1 additions (rules-engine)
* **Multiclassing** (`CharacterBuild.classes = ((class_id, level, subclass), ...)`, first entry = primary; `level` = total).
  Ability prerequisites (SRD table, `data.classes.MULTICLASS_PREREQS`) are enforced for *every* class once there are two
  or more (`make_pc` raises `ValueError`; `multiclass_problems` lists violations). Saving throws and starting skills come
  from the first class; later classes grant only `MULTICLASS_PROFICIENCIES` (armor/weapons, plus one skill for bard/ranger/rogue
  via `choices['mc_skills']`). HP: first level of the first class is the maximum die, then each class's die; hit dice are kept
  per die size (`fstate['hd_left'/'hd_max']`, largest die spent first, long rest regains half the total, largest first).
  Proficiency bonus uses total level. Features follow class levels; a feature id shared by classes (Extra Attack, ASI) is
  granted once, so Extra Attack does not stack; Unarmored Defense variants are AC bases and the best wins. Resources with the
  same id (Channel Divinity) form one pool whose size is the sum of the grants. Spell slots: one spellcasting class -> its own
  table; several -> combined Spellcaster level (full + half rounded down; no third-casters in this content set) on the
  full-caster table; Warlock Pact Magic is separate, and pact and ordinary slots pay for each other's spells. Spells are
  per class (`choices['class_spells'|'class_cantrips']`) and each spell uses the casting ability of its class
  (`fstate['spell_abil']`, swapped in `cast_spell`). Concentration/zone DCs use the casting ability active when they were created.
  Archetypes: `paladin_sorcerer`, `fighter_wizard`, `rogue_fighter`, `cleric_druid` (total level 2-5) and
  `random_build(..., multiclass_prob=p)`.
* **Carrying capacity.** STR x 15 lb (Tiny x0.5, Large x2, Huge x4, Gargantuan x8); push/drag/lift x2. Weights come from
  item/weapon/armor defs; 50 coins = 1 lb. `RulesConfig.encumbrance`: `"capacity"` (default; over capacity -> speed 5 ft),
  `"variant"` (> 5 x STR -10 ft; > 10 x STR -20 ft plus disadvantage on attacks, STR/DEX/CON checks and saves), `"off"`.
  Picking up loot that over-encumbers is always allowed. Heavy armor with unmet STR still costs 10 ft.
* **Attunement.** `ItemDef.attunement` items work only while `ItemInstance.attuned` (AC/save bonuses, worn effects,
  wand/item abilities). Max 3 per creature. Items carried at spawn are attuned (worn/equipped first, then most valuable);
  every short rest fills free slots with the most valuable unattuned items; `World.attune/unattune/auto_attune` are the API
  (no action slot: the action space is unchanged). Transfers clear attunement. `RulesConfig.attunement=False` disables the rule.
* **Stable creatures** regain 1 HP after 1d4 hours: the clock starts (rolls 1d4) the first time time passes (a rest) after
  stabilising, ticks in rest rounds (1 h = 600 rounds), is dropped on damage / re-stabilising / healing.
* **Underwater combat.** Attackers in deep water without a swim speed have disadvantage with melee weapons other than
  dagger/javelin/shortsword/spear/trident; ranged weapon attacks beyond normal range auto-miss and have disadvantage
  within it unless crossbow/net/javelin/spear/trident/dart; creatures in deep water resist fire. Spells are unaffected.
* **Projectile / gas barriers.** Zones with `params['blocks_projectiles']` (Wind Wall), zone implementations with
  `blocks_projectile(world, z, a, b)` and target `projectile_block` hooks make ordinary ranged weapon attacks that cross
  them miss (segment-vs-cell test between square centres). Gas/fog zones (`fog_cloud`, `stinking_cloud`, `params['gas']`) do not
  spread across `blocks_gas` zones. Sealed exits block flyers too and the render adapter reports `state='sealed'`.
* **Climbing.** Low walls (`Terrain.LOW_WALL`) no longer block movement: crossing costs 10 ft per square (difficult terrain)
  or 5 ft with a climb speed; they still give half cover. Statues, pillars and walls still block. Spider Climb
  creatures simply have the climb speed (no ceilings).
* **Burrowing.** Creatures with a burrow speed use DESCEND at z=0 to dig in (5 ft) and ASCEND to surface; `z=-1` means
  burrowed: not on the ground occupancy grid, unseen except by tremorsense (both ways), immune to area effects, cannot target
  others, tunnels only through diggable ground (dirt/sand/grass/cave floor/rubble/difficult terrain; stone floors and walls are
  not) at `walk/burrow` x 5 ft per square; `earth_glide` creatures tunnel through stone and walls but never water/lava/pits.
  Emerging needs a free walkable square. Stone dungeons are therefore mostly burrow-proof, as in the SRD.
* **Mounted combat** (`RulesConfig.mounts`). Monsters tagged `mount` (warhorse, riding/draft horse, pony, mule, elk, giant elk,
  giant eagle, griffon, hippogriff) can be ridden by an allied creature at least one size smaller within 5 ft via the
  `mount` / `dismount` abilities (specific slots, appended last; they appear only in worlds that contain a mount; cost half the
  rider's speed, no action). `mount_independent` mounts (giant elk, giant eagle) keep their own turns and AI and the rider
  cannot move; all others are *controlled*: their turn is merged into the rider's, the rider's MOVE/APPROACH/ASCEND move the
  mount using the mount's speed (the rider's pool mirrors the mount's; the rider's own pool is parked and restored on
  dismount). Dash/Disengage/Dodge of the rider also apply to the controlled mount (deviation: they spend the rider's
  action, not the mount's). The rider shares the mount's square (no ground occupancy of its own; distance to a rider is
  measured from the mount's body). Opportunity attacks against a moving mount are aimed at the weaker of rider and mount. A mount
  knocked prone makes the rider step off (reaction, land on feet) or fall prone; a prone rider needs a DEX save DC 10 to stay
  seated; a mount moved against its will (forced movement, teleport) forces the same save; a downed/dead mount throws the rider
  prone. Scenarios seat riders with the spawn tag `ride:<tag>` on the nearest allied mount carrying `<tag>` (`mounted_patrol`).
  Events `MOUNT=74`, `DISMOUNT=75`; `EntityView.mounted_on`.
* **Hooks / API.** `DecisionKind.POST_ROLL=12`, hooks `observe_damage`, `observe_hit_die`, zone `on_damage`; `projectile_block`
  now consulted. New `RulesConfig` fields `encumbrance`, `attunement`, `mounts`. `ItemInstance.penalty` (corrosion) is honoured
  in weapon damage; `ignores_difficult` and `may_share_space` (pass 2) cover per-creature terrain immunity and incorporeal /
  swarm / engulfing creatures sharing squares. Turn-undead auras of allies (ghast) are content-side via `ally_save_mods`.
