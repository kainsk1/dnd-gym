# Scenario library

Builder: `emberdeep.scenarios.builder.build_world(scenario, seed=0, rules_overrides=None, **scenario_kwargs) -> (World, BuildInfo)`.
Registry: `registry.register_scenario(id)`, `get_scenario(id, **kw)`, `list_scenarios()`, `scenario_info(id)`.
Game-theoretic rationale of every scenario: `docs/SCENARIO_DESIGN.md`; maps: `docs/MAPS.md`.

## Builder

`ScenarioSpec` -> world: map (library `.txt`, inline ASCII or `mapgen` generator; map seed fixed unless
`randomize["map_seed"]`) -> factions/relations -> objects (`ObjectSpec` configures the map's existing chest/loot/altar/door at
the location, or creates one; `at` = `(x, y)`, `"@marker"` or `"region:<name>"` = every matching map object in the region) ->
spawns -> objectives -> `ScenarioRuntime` (`world.scenario_runtime`) -> `world.start()` -> `runtime.after_step(world)`.

* **Spawns.** `at`: `"@x"` marker, `(x, y)`, list of candidate cells, `"region:<name>"`. Groups (`count`) take distinct cells
  (random subset when there are more candidates than creatures and `jitter`, spill to the nearest free squares otherwise).
  Templates: `monster:<id>`, `pc:<archetype>[/<race>]`, `pc:random`, `build:<key>`. `agent=True` makes an RL agent
  (`agent_id` default `"<faction>_<i>"`, ids stable in spawn order). PCs without darkvision carry a lit torch on `dark`/`dim` maps.
  `hp_fraction`, `downed`, `gold`, `items`, `light` as in `SpawnSpec`.
* **Tags** (all kept on `creature.tags` for the scripted AI): `asleep` (UNCONSCIOUS with ends-on-damage; woken by damage, a
  hostile adjacent / near (50%/round within `wake_radius`), combat noise within hearing radius, `wake:<region>` intruders,
  or object triggers; waking removes the tag), `hidden` (Stealth rolled -> `HIDDEN`, `stealth_total`), `slow` (speed x0.5),
  `guard:<region>`, `stationary`, `vip`, `boss`, `captain`, ... (read by `emberdeep.ai` / objectives / runtime).
* **Domain randomisation** (`spec.randomize`): `party: "random_classes"` (agent PCs get random class/race/scores), `party_level: [lo, hi]`,
  `spawn_jitter: r` (move each spawn to a random free cell within r), `map_seed: True`.
  Pass through the factory's `randomize=` knob or `build_world(..., randomize={...})`.
* **Rules.** `spec.rules` < `rules_overrides`; `max_rounds`/`max_steps` default to the spec's.
* **Procedural population.** `spec.runtime["populate"] = "module:fn"` is called as `fn(map_data, spec, rng) -> (spawns, objects)`
  (used by `deep_delve`); `spec.runtime["generators"]` picks the map generator per episode.

## Runtime (`scenarios/runtime.py`)

`runtime.after_step(world)` after every `world.step` (idempotent; already called once by the build) and **before** draining
events. It handles waves (`at_round`/`every`/`repeat`/`scale_with_round`/`trigger_region`/`trigger_item`; arrivals emit `SUMMON`
(src -1) + a `LOG` announce, join the turn order, are alerted (everyone is aware of everyone, combat starts) so scripted
hunters attack), sleepers, region occupancy, the special rules in `spec.runtime` (`escort`, `bribe`, `last_standing`,
`relic_escape`) and event counters (truces, betrayals, PvP damage, boss damage, loot by region).
`runtime.terminal_payoffs(world) -> {agent_id: {component: reward_units}}` (components `escort_fee`, `prize`, `survival`;
the generic `terminal_payoff` gold modes stay in the env). `runtime.metrics(world) -> {name: float}` (see per scenario).

The runtime is immutable configuration shared by `World.clone`; all state is in `world.stats["scn:*"]` (scalars/tuples) and in
altar `state` dicts, so clones evolve independently (tested).

## Scenario objects (`rules/content/objects_scenario.py`)

| hook | params | behaviour |
|---|---|---|
| `scn_fountain` | `charges` 6, `capacity` K 8, `regen_r` 0.25, `heal` "2d8+2" | drink = -1 charge + heal; each round `c += r c (1 - c/K)`; exhausted never regrows |
| `scn_shrine` | `threshold` 200, `deadline_round` 6, `chunk` 25, `refund`, `effect`, `ward_ac` | donate gold; threshold before deadline -> donor-faction members get `ward_of_the_shrine` (+AC, necrotic resistance) |
| `scn_winch` | `required` 3, `targets` "exit" or "region:<n>" | cumulative interactions unseal the exits (`ExitMarker.present`) |
| `scn_relic` | `item`, `awaken` tags | take the relic item |
| `scn_coffer` | `lots` 10, `lot_gp` 100, `guardian_tag` | each use pays a lot, applies `cursed_marker`, first use wakes the guardians |

Effects: `ward_of_the_shrine`, `cursed_marker`.

## Scenarios

Agents are `<faction>_<i>` unless noted. "gold_if_escaped" etc. are `RewardSpec.terminal_payoff` modes paid by the env.

### goblin_ambush (+ `_team`, `_small`) - pure cooperation
Map `goblin_ambush`; 4 agents (fighter, cleric, rogue, wizard, L2) vs hidden goblins in four brush patches, goblin archers, a bugbear
chief. Objective DEFEAT_ALL (+1). Knobs: `party`, `level`, `n_goblins`, `n_archers`, `bugbear`, `hidden`, `team_reward`, `max_rounds`.
`_team`: `team_reward=True`. `_small`: 2 agents (fighter, cleric) vs 3 goblins on an inline 20x14 map, starts in combat.
Metrics: common ones (`agents_alive`, `agents_dead`, `rounds`...).

### dragon_hoard - stag hunt
Map `dragon_hoard`; 4 L4 agents. Sleeping red dragon wyrmling (`boss`, wakes on entering `dragon_chamber`) + a hell hound on a
3,000 gp hoard (3 chests, 8 piles); two kobold-guarded alcove chests (150 gp), dozing sentries, chamber guards.
`gold_if_escaped` (scale 0.001), no victory. Scripted-AI check (L4 party, hounds=1, 30 seeds): dragon killed by 1 hero 0%, 2 heroes ~10%,
3 heroes ~57%, 4 heroes ~70%. Knobs: `party`, `level`, `hoard_gp`, `alcove_gp`, `dragon_asleep`, `kobolds`, `hounds`, `terminal_scale`.
Metrics: `boss_all_engaged`, `boss_engagers`, `boss_damage/<a>`, `bosses_dead`, `loot_agents/<region>`, `loot/<region>/<a>`, `wealth_gini`.

### cursed_coffer - prisoner's dilemma / chicken
Map `cursed_coffer`; factions `west` (fighter, rogue; locked door) and `east` (paladin, rogue), NEUTRAL, 2 agents each (`west_0`...).
Coffer altar (10 x 100 gp lots) wakes 4 animated-armor guardians; takers get `cursed_marker`. `gold_if_escaped`.
Knobs: `west`, `east`, `level`, `lots`, `lot_gp`, `guardians_per_side`. Metrics: `coffer/taken/<a>`, `coffer/take_gini`, `coffer/lots_left`,
`truces`, `betrayals`, `first_truce_round`, `first_betrayal_round`, `pvp_damage`, `occupancy/dais/<a>`.

### rival_delvers - coopetition
Map `rival_delvers`; `west`/`east` parties of 3 (NEUTRAL). Monster rooms (goblins), relic guardians (ogre + 2 orcs, `guard:relic_room`),
relic altar. First faction to ESCAPE carrying the Ember Relic wins (placeholder objectives `relic_<faction>` completed by the runtime).
Knobs: `west`, `east`, `level`, `room_monsters`, `room_monster`, `chest_gp`. Metrics: `relic/taken`, `relic/taken_by_faction`, `relic/taken_round`,
`truces`, `betrayals`, `pvp_damage`, `bosses_dead`.

### fountain_of_life - common pool
Map `fountain_of_life`; 4 L3 agents. Fountain altar (logistic regrowth); goblin waves every 3 rounds (+1 per ~3 repetitions), skeleton waves every 6.
Objective SURVIVE 18 (+1). Knobs: `charges`, `capacity`, `regen_r`, `heal`, `duration`, `wave_every`, `goblins_per_gate`, `wave_scale`, `skeletons`.
Metrics: `fountain/charges`, `fountain/level`, `fountain/mean_level`, `fountain/uses`, `fountain/uses/<a>`, `fountain/use_gini`,
`fountain/heal_total`, `fountain/depletion_round`, `wave_spawns`.

### shrine_of_wards - threshold public good
Map `shrine_of_wards`; 4 L3 agents with 100 gp each. Shrine altar (200 gp before round 6 -> ward). Undead waves from round 6, a wight at round 12.
`gold_if_alive` (scale 0.002), SURVIVE 16 (+1). Scripted check: the ward halves deaths. Knobs: `start_gold`, `threshold`, `chunk`, `deadline`, `refund`,
`ward_ac`, `duration`, `wave_start`, `wave_every`, `undead_per_gate`, `open_chapels`.
Metrics: `shrine/pool`, `shrine/warded`, `shrine/ward_round`, `shrine/donors`, `shrine/free_riders`, `shrine/free_rider_index`,
`shrine/donated/<a>`, `shrine/donation_gini`.

### hold_the_bridge - volunteer's dilemma
Map `hold_the_bridge`; 4 L3 agents with 200 gp. Winch altar (3 cranks) unseals the exit; four raider waves cross the 1-wide bridge.
`gold_if_escaped` (scale 0.002), ESCAPE (all remaining) +0.5. Knobs: `winch_required`, `start_gold`, `n_waves`, `first_wave`, `wave_gap`.
Metrics: `occupancy/bridge/<a>`, `occupancy_max/bridge`, `winch/cranks/<a>`, `winch/opened_round`, `agents_escaped`, `agents_dead`.

### gladiator_pit - free-for-all with alliances
Map `gladiator_pit`; 6 agents (`gladiator_0..5`, factions `g0..g5`, all HOSTILE, random classes L3). Manticore at round 5. The episode ends when all
surviving gladiator factions are mutually TRUCE/ALLIED and the manticore is dead; prize 1,000 gp split among survivors (`prize`), 0.02 per
round survived (`survival`), objective `last_standing` (+1) for the coalition. Knobs: `n_agents`, `level`, `manticore_round`, `prize`, `survival_bonus`.
Metrics: `truces`, `betrayals`, `first_betrayal_round`, `pvp_damage`, `boss_engagers`, `alive/<a>`.

### caravan_escort - protection public good + bargaining
Map `caravan_escort`; 4 L3 agents (60 gp each) escort the merchant (`vip`, `slow`, ai `vip`, ALLIED `caravan` faction). Bandits on six ledges, a toll line and Captain Varga
(`captain`). >= 100 gp given to the captain by party members -> TRUCE with the bandits. VIP arrival (region `goal` or escape) wins (objective `vip_safe`)
and pays `escort_fee` 150 gp (x scale) to surviving agents; VIP death fails. Sight radius 14. Knobs: `bribe`, `escort_fee`, `start_gold`, `bandits_per_ledge`, `toll_bandits`.
Metrics: `bribe/<a>`, `bribe/total`, `bribe/paid`, `vip_arrived`, `vip_arrival_round`, `vip_alive`.

### deep_delve - procedural
`bsp_dungeon` | `cave` | `crypt` per episode (map seed = episode seed), random-class party of 4 at one random level 1-5, themed encounters per room via
`build_encounter` (some asleep, none within `min_start_dist` of the entrance), boss room with a `boss`, loot-table chests, mapgen traps and keyed
locked doors (key items are registered on the fly). `gold_if_escaped`, objective `slay_the_boss` (+1, does not end the episode). Knobs: `n_agents`,
`level_range`, `generators`, `difficulties`, `room_prob`, `asleep_prob`, `max_monsters`, `boss`, `boss_difficulty`, `min_start_dist`, `sight_radius`, `width`, `height`.
Metrics: `bosses_dead`, `boss_engagers`, `wealth_gini`, `agents_escaped`.

### duel / skirmish_2v2 - micro
Inline walled arena (`duel.arena_rows`), random classes L3, start in combat, zero-sum DEFEAT_ALL objectives per side. `duel`: 1v1 (`red_0`, `blue_0`);
`skirmish_2v2`: 2v2. Knobs: `level`, `max_rounds`.

## Common metrics
`rounds`, `agents_alive`, `agents_dead`, `agents_escaped`, `wealth_gini`, `gold_total`, `waves_fired`, `wave_spawns`, `truces`, `betrayals`,
`first_truce_round`, `first_betrayal_round`, `pvp_damage`, `asleep_now`, `alive/<agent>`, plus `bosses_*`/`boss_*` when a creature has the `boss` tag.
