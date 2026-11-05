# Class features and racial traits (levels 1-5)

Implementation: `emberdeep/rules/content/class_*.py`, `class_common.py`, `feature_effects.py`, `race_traits.py`.
Tests: `tests/test_features_<class>.py`, `tests/test_race_traits.py`, `tests/test_archetypes_play.py`.
Engine requests that would make deviations exact: `docs/dev/contract_requests/features.md` (referred to as **CR n**).

## Conventions

* Every active feature is a `FeatureAbility` (`kind="feature"`) with the right cost (ACTION / BONUS_ACTION / FREE /
  reaction decision), an automatic class-resource charge (`resource`/`n`: rage, ki, channel_divinity, ...) and a
  `vfx` key from `render/api.VFX_KEYS`. Resources live in `Creature.resources` and recharge in `World.short_rest` /
  `long_rest`; per-rest flags (Natural Recovery, the once-per-rest invocations) live in the hook `state` and reset in
  `on_long_rest`.
* Abilities are always *listed* and masked by `can_use` (never added/removed mid-turn), the exception being Wild Shape,
  whose forms replace weapon/spell abilities while transformed.
* Reactions and on-hit riders are decisions (`ON_ATTACKED`, `ON_DAMAGED`, `ALLY_ATTACKED`, `ON_HIT_RIDER`); the default auto
  decider answers them for scripted creatures. Option keys: `uncanny_dodge`, `deflect_missiles`, `cutting_words`,
  `stunning_strike`, `open_hand_prone|push|no_reactions`, `giant_killer`, `smite_N`.
* **Build choices** (`CharacterBuild.choices`) reach the features through `class_common.choices_of(creature)` (`c.build`,
  see CR 1). Container features (`metamagic`, `eldritch_invocations`, `pact_boon`, `hunters_prey`) attach the chosen option
  features (they are separate registered ids) the first time abilities are built; `InitFeature`s perform one-time
  setup (extra skills, cantrips, senses) the same way (CR 2).
* "Once per turn" limits use the key `(round, active creature)` so they also hold on other creatures' turns.
* Bonus-action attacks (Frenzy, Horde Breaker) use `class_common.bonus_attack`, which runs a weapon ability without
  spending the Attack action.
* Deterministic: all randomness is `world.rng`. `peek_rng` copies the generator to *predict* (never consume) dice.

## Barbarian

| id | implementation | deviation |
|---|---|---|
| `rage` | Bonus action, resource `rage`, not in heavy armor. Effect `rage` (10 rounds): +2 (+3 at 9th) damage on STR-based melee weapon/unarmed attacks, resistance to B/P/S (also magical), advantage on STR checks/saves, ends when the bearer's turn ends without attacking a hostile or taking damage, or on unconscious; sets `S.RAGING`; ends concentration. | Casting spells while raging is not blocked (barbarians have no spells). |
| `unarmored_defense_barbarian` | exemplar (`rules/features.py`). | |
| `reckless_attack` | Free toggle before the first attack of the turn: effect `reckless` until the start of the next turn: advantage on STR melee attacks, attacks against the bearer have advantage. | |
| `danger_sense` | Advantage on DEX saves unless blinded, deafened or incapacitated. | The "effects you can see" clause is not modelled (all DEX saves). |
| `fast_movement` | `speed_bonus` +10 ft unless wearing heavy armor. | |
| `frenzy` | Berserker: bonus-action melee attack while raging (`bonus_attack`); the rage ends with +1 exhaustion if Frenzy was used during it. | Frenzy is "chosen" by using the ability, not when starting the rage. |
| `extra_attack` | exemplar. | |

## Bard

| id | implementation | deviation |
|---|---|---|
| `bardic_inspiration` | Bonus action on an ally (not self) within 60 ft, resource `bardic_inspiration` (CHA mod, long rest); effect `bardic_inspiration_die` (d6/d8, 100 rounds). | See next row. |
| (die usage) | The holder's attack rolls and saving throws: the next d20 is predicted from an RNG copy; the die is added (and consumed) only when it turns a failure into a success. | Not applied to ability checks; automatic instead of a player choice (CR 8). |
| `jack_of_all_trades` | +half proficiency on non-proficient ability checks (also passive) and initiative. | |
| `song_of_rest` | `on_short_rest`: every wounded ally (and self) heals an extra d6 (d8 at 9th). | Not tied to spending Hit Dice. |
| `font_of_inspiration` | 5th level: Bardic Inspiration uses return on a short rest. | |
| `expertise_bard` | Factory applies `build.expertise`; marker. | |
| `bonus_proficiencies_lore` | `InitFeature`: three skills from `choices['lore_skills']`. | |
| `cutting_words` | Reaction (resource `bardic_inspiration`): `-d6/d8` on an enemy attack roll against an adjacent ally (`ALLY_ATTACKED`, before the roll) or against the bard (`ON_ATTACKED`, after the roll; offered only when it can turn the hit into a miss, via `ac_bonus`). Enemy within 60 ft and visible. | Attack rolls only; allies must be adjacent to the bard (engine's `protectors`). |

## Cleric

| id | implementation | deviation |
|---|---|---|
| `channel_divinity` | Marker; uses are the `channel_divinity` resource (short rest). | |
| `turn_undead` | Action: undead within 30 ft that see the cleric make a WIS save vs spell DC; failure: effect `turned` (10 rounds, no reactions, ends on damage) + FRIGHTENED. | "Turned" is frightened + no reactions (no forced Dash); undead only. |
| `destroy_undead` | Companion of Turn Undead: failed-save undead with CR <= 1/2 (table by level) are destroyed. | |
| `preserve_life` | Life domain: action, pool 5 x level shared out (most wounded first) among allies within 30 ft, each up to half max HP; not undead/constructs. | |
| `disciple_of_life` | `on_cast`: each target of a leveled healing spell heals an extra 2 + slot level. | |
| `bonus_proficiency_life` | Heavy armor proficiency (`armor_profs`). | Armor proficiency is not enforced by the engine. |
| `divine_domain` (`subclass_marker`) | Marker. Domain spells are attached by the factory (verified: always prepared). | |

## Druid

| id | implementation | deviation |
|---|---|---|
| `wild_shape` | Action, resource `wild_shape` (2, short rest). Three form abilities (hardest-hitting, toughest, fastest legal beast) + `revert_wild_shape` (bonus action). Transform: STR/DEX/CON, speeds, size, type (beast), senses, resistances, HP pool, AC (equipment merges), natural attacks and multiattack replace the druid's; INT/WIS/CHA, proficiencies, hooks, concentration are kept; no spells. Revert at 0 HP with overflow damage carried over, voluntarily, or when the duration (hours = level/2) ends. Beast traits become hooks. | Max CR 1/4, 1/2 at 4th; swimming from 4th, flying from 8th. Massive damage uses the beast's max HP (CR 7). Large forms need a free 2x2 footprint. |
| `natural_recovery` | Land: on a short rest recover slots of combined level <= ceil(level/2), none above 5th, once per long rest. | Automatic (no choice of slots; highest first). |
| `druidic`, `circle_spells`, `bonus_cantrip_land`, `druid_circle` | Markers; circle spells and the land cantrip come from the factory (verified). | |

## Fighter

| id | implementation | deviation |
|---|---|---|
| `fighting_style` | Marker; the chosen style is attached by the factory. | |
| `archery`, `defense`, `dueling`, `protection`, `two_weapon_fighting` | exemplars. | |
| `great_weapon_fighting` | Two-handed / versatile-in-two-hands melee: the first damage component's dice are pre-rolled with 1s and 2s rerolled once (crit aware). | |
| `second_wind` | exemplar; short-rest resource verified. | |
| `action_surge` | Free, resource `action_surge`: resets Action + Attack-action bookkeeping. | Only after the Action is spent and all attacks made. |
| `improved_critical` | Champion: `crit_min` 19 on weapon/unarmed attacks. | |
| `extra_attack` | exemplar (tested at 5th). | |

## Monk

| id | implementation | deviation |
|---|---|---|
| `unarmored_defense_monk` | exemplar (`rules/features.py`): AC 10 + DEX + WIS, no armor and no shield. | |
| `martial_arts` | Re-registered subclass of the exemplar: die by level plus the bonus-action unarmed strike after the Attack action (`martial_arts_bonus`). | DEX for monk weapons only for finesse weapons (CR 5). |
| `ki` | Marker; resource `ki` (monk level, short rest). | |
| `flurry_of_blows` | Bonus action, 1 ki, after the Attack action: two unarmed strikes (second retargets if the first target is gone). | |
| `patient_defense` | Bonus action, 1 ki: Dodge. | |
| `step_of_the_wind` | Bonus action, 1 ki: `step_disengage` / `step_dash`. | Jump distance not modelled. |
| `unarmored_movement` | Speed bonus by level while unarmored and without shield. | Wall/water running not modelled. |
| `deflect_missiles` | Reaction when hit by a ranged weapon attack: `-(1d10 + DEX + level)`; if reduced to 0 and 1 ki is available the missile is thrown back (ranged attack with monk proficiency). | |
| `slow_fall` | Reduces `fall` damage by 5 x level (uses the reaction). | |
| `stunning_strike` | Rider after a melee hit: 1 ki, CON save vs 8 + prof + WIS, stunned until the end of the monk's next turn (`expire="end_source"`). | |
| `open_hand_technique` | Rider on Flurry hits: prone (DEX), push 15 ft (STR) or no reactions until the end of the monk's next turn. | |
| `extra_attack` | exemplar. | |

## Paladin

| id | implementation | deviation |
|---|---|---|
| `divine_sense` | Action, resource `divine_sense`: celestials/fiends/undead within 60 ft (not behind total cover) are revealed (`hit_history`, `aware`, `SPOTTED`). | |
| `lay_on_hands` | Action, touch: heals min(pool, missing HP) from the `lay_on_hands` pool (revives at 0 HP); `lay_on_hands_cure` spends 5 to remove poisoned. Not undead/constructs. | Heals as much as needed (no partial amounts). |
| `divine_smite` | exemplar (verified: slot cost, 2d8 +1d8/level, +1d8 vs undead/fiends, crit doubling, rider decision `smite_N`). | |
| `divine_health` | Marker. | The engine has no disease. |
| `sacred_weapon` | Action, channel: effect `sacred_weapon` (10 rounds) +CHA (min 1) to weapon attack rolls, attacks magical; the bearer sheds light (20 ft bright + 20 ft dim) while it lasts. | |
| `turn_the_unholy` | Like Turn Undead for fiends and undead (no destruction). | |
| `channel_divinity`, `fighting_style`, `sacred_oath` | Markers; oath spells from the factory (verified). | |

## Ranger

| id | implementation | deviation |
|---|---|---|
| `favored_enemy` | Advantage on Survival and INT checks naming a `target` of the chosen type (`choices['favored_enemy']`). | Exploration only. |
| `natural_explorer` | Marker. | Travel benefits are not modelled. |
| `primeval_awareness` | Action: spends the lowest spell slot; reports creature types present (event `found`). | Locations are not revealed. |
| `hunters_prey` | Container for the chosen option. | |
| `colossus_slayer` | +1d8 once per turn on a weapon hit against a creature below max HP. | |
| `giant_killer` | Reaction (`ON_DAMAGED`) when a Large+ creature within 5 ft hits you: melee attack against it. | Only after hits. |
| `horde_breaker` | Once per turn, a weapon attack triggers a second attack with the same weapon on another hostile adjacent to the target (in weapon range) via `bonus_attack`. | |

## Rogue

| id | implementation | deviation |
|---|---|---|
| `sneak_attack` | Re-registered (supersedes the exemplar): finesse/ranged weapon, advantage or an able ally adjacent to the target and no disadvantage; once per turn of *any* creature; ally check uses the full incapacitation mask. | |
| `expertise_rogue`, `thieves_cant` | Factory / flavour markers. | |
| `cunning_action` | Three bonus-action abilities: `cunning_dash`, `cunning_disengage`, `cunning_hide`. | |
| `uncanny_dodge` | exemplar (verified: halves damage of a seen attacker). | |
| `fast_hands` | Thief: `fast_hands_interact` (use an object) and `fast_hands_pick_lock` as bonus actions. | Sleight of Hand not modelled. |
| `second_story_work` | Marker. | Climbing is not modelled. |

## Sorcerer

| id | implementation | deviation |
|---|---|---|
| `font_of_magic` | Marker; resource `sorcery_points` (level, long rest). | |
| `flexible_casting` | Bonus action: `create_slot_1..3` (2/3/5 SP) and `convert_slot` (lowest slot that fits under the SP maximum). | Created slots may exceed the normal maximum until a long rest. |
| `metamagic` | Container; each chosen option grants a FREE toggle that arms it for the next spell cast this turn; points are paid and the option applied when the spell is cast (`on_cast`). | |
| `empowered_spell` | Spell damage dice pre-rolled; up to CHA-mod lowest dice below average rerolled. | |
| `extended_spell` | Doubles `remaining` of effects/conditions/zones the spell created. | |
| `quickened_spell` | Arm before using the Action; the spell then costs the bonus action. | Cannot be armed after the Action is spent (CR 4). |
| `twinned_spell` | Single-target spell (at the cast slot) is cast again for free on the nearest enemy / most wounded ally; concentration is shared. | |
| `careful_spell` | Up to CHA-mod allies in the area take half damage (none if no half clause). | Conditions not prevented, no save roll (CR 3). |
| `heightened_spell` | Implemented via `force_save_mods`. | Not offered until CR 3 exists. |
| `distant_spell`, `subtle_spell` | Registered. | Not offered (CR 4). |
| `dragon_ancestor` | Ancestry in `choices`; double proficiency on CHA checks vs dragons. | |
| `draconic_resilience` | +1 HP/level (factory `hp_per_level`), unarmored AC 13 + DEX. | |

## Warlock

| id | implementation | deviation |
|---|---|---|
| `pact_magic` | Marker; pact slots from the factory, recharge in `short_rest` (verified). | |
| `dark_ones_blessing` | `on_kill`: temp HP = CHA mod + level (min 1). | Only when the hostile dies (not for downed PC-like creatures). |
| `eldritch_invocations` | Container (level and pact prerequisites enforced). | |
| `agonizing_blast` | +CHA to each Eldritch Blast beam. | |
| `repelling_blast` | A Large-or-smaller creature hit by Eldritch Blast is pushed 10 ft. | |
| `armor_of_shadows` | Permanent unarmored AC 13 + DEX. | Equivalent to at-will Mage Armor. |
| `devils_sight` | `senses['devils_sight']=120` and darkvision 120. | Does not pierce magical darkness (CR 9). |
| `eldritch_sight` | Detect magic at will (`innate = -1`). | |
| `fiendish_vigor` | Action: 8 temporary HP. | |
| `beguiling_influence` | Deception + Persuasion proficiency. | |
| `one_with_shadows` | Action in dim light/darkness: invisible until you move or attack. | |
| `thirsting_blade` | Two attacks per Attack action with the pact of the blade. | |
| `thief_of_five_fates`, `mire_the_mind`, `sign_of_ill_omen` | Once per long rest cast bane / slow / bestow curse with a pact slot. | |
| `beast_speech`, `book_of_ancient_secrets`, `eyes_of_the_rune_keeper`, `gaze_of_two_minds`, `mask_of_many_faces`, `misty_visions`, `voice_of_the_chain_master` | Registered, inert. | Spells outside the engine list / flavour. |
| `eldritch_spear` | Registered, inert. | Needs CR 4. |
| `pact_boon` | Container. | |
| `pact_of_the_blade` | Action: magical finesse "Pact Blade" (rapier stats) equipped, proficient. | |
| `pact_of_the_chain` | Action: summons the familiar (`choices['familiar']`, default imp) as a scripted ally in the initiative order. | |
| `pact_of_the_tome` | Adds `choices['tome_cantrips']`. | |

## Wizard

| id | implementation | deviation |
|---|---|---|
| `arcane_recovery` | Short rest, once per long rest (resource): slots of combined level <= ceil(level/2), none above 5th, highest first. | |
| `sculpt_spells` | Up to 1 + spell level allies take no damage from your evocation area spells. | Conditions are not prevented (CR 3). |
| `evocation_savant`, `arcane_tradition` | Markers. | |

## Common

`ability_score_improvement` (applied by the factory), `subclass_marker` for every subclass choice, `fighting_style`.

## Racial traits

| id | implementation | deviation |
|---|---|---|
| `darkvision`, `dwarven_combat_training`, `dwarven_toughness`, `elf_weapon_training`, `hellish_resistance`, `high_elf_cantrip`, `infernal_legacy`, `menacing` | Registered inert traits: the static effect (senses, proficiencies, HP, resistance, cantrip, innate spells, skill) is applied by the factory from `RaceDef` (verified in `test_race_traits`). | |
| `artificers_lore`, `stonecunning`, `tinker`, `trance` | Registered inert. | Lore/downtime/rest flavour. |
| `draconic_ancestry` | Marker (`choices['draconic_ancestry']`). | |
| `breath_weapon` | `MonsterSave` ability: cone/line by ancestry, DC 8 + CON + prof, damage by level, half on success, 1 use per short rest (`uses_left`). | |
| `damage_resistance` | Resistance to the ancestry damage type. | |
| `dwarf_speed` | +10 ft when Strength is below the armor requirement (cancels the penalty). | |
| `naturally_stealthy` | `can_hide`: a creature at least one size larger adjacent to the hider and nearer the observer. | |
| `halfling_nimbleness` | Registered inert. | CR 6. |
| `skill_versatility` | `InitFeature`: two skills from `choices['racial_skills']`. | |
| `brave`, `dwarven_resilience`, `fey_ancestry`, `gnome_cunning`, `keen_senses`, `lucky`, `relentless_endurance`, `savage_attacks` | exemplars in `rules/traits.py` (not re-implemented). | |
