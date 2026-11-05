# Emberdeep RL API (`emberdeep.env`)

Emberdeep exposes tactical 5e SRD combat / exploration as **PettingZoo** (AEC and Parallel), **Gymnasium**
(single agent) and **batched vector** environments. All of them are thin wrappers around one engine-facing
core (`emberdeep.env.core.EmberdeepCore`), so observations, masks, rewards and metrics are identical everywhere.

* [Quickstart](#quickstart)
* [Entry points and `EnvConfig`](#entry-points-and-envconfig)
* [Agents and turn semantics](#agents-and-turn-semantics)
* [Action space](#action-space)
* [Observation](#observation)
* [Global state (CTDE)](#global-state-ctde)
* [Rewards](#rewards)
* [Metrics](#metrics)
* [Vector environments](#vector-environments)
* [Rendering, replays, human play](#rendering-replays-human-play)
* [Determinism, cloning, seeding](#determinism-cloning-seeding)
* [Performance](#performance)
* [Testing and known limitations](#testing-and-known-limitations)

---------------------------------------------------------------------------------------------------------------

## Quickstart

```python
import numpy as np
import emberdeep

print(emberdeep.list_scenarios())

# ---- PettingZoo AEC ---------------------------------------------------------------------------------------
env = emberdeep.make_aec("goblin_ambush")                   # keyword arguments = EnvConfig fields
env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()   # obs is a Dict of arrays, incl. obs["action_mask"]
    if terminated or truncated:
        env.step(None)                                      # PettingZoo dead step
        continue
    env.step(int(np.random.choice(np.flatnonzero(obs["action_mask"]))))
print(env.episode_metrics()["global"])

# ---- Parallel: all agents submit an action, only the decider's is applied ----------------------------------
penv = emberdeep.make_parallel("dragon_hoard", prosocial=0.5)
obs, infos = penv.reset(seed=0)
while penv.agents:
    actions = {a: int(np.random.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in penv.agents}
    obs, rewards, terminations, truncations, infos = penv.step(actions)

# ---- Gymnasium: control one agent, hero bots play the others -------------------------------------------------
genv = emberdeep.make_gym("goblin_ambush", agent="party_0", others="scripted")

# ---- Vectorised (subprocesses + shared memory), batched arrays [num_envs, n_agents, ...] --------------------
venv = emberdeep.make_vector("goblin_ambush", num_envs=8)
obs, infos = venv.reset(seed=0)
obs, rewards, terminations, truncations, infos = venv.step(np.zeros((8, venv.n_agents), np.int64))
venv.close()
```

Runnable examples: `examples/random_rollout.py`, `examples/pettingzoo_quickstart.py` (a NumPy baseline policy
that wins `goblin_ambush_small` 5 times out of 6 using nothing but the observation).

Command line tools: `python -m emberdeep.play`, `python -m emberdeep.viewer`, `python -m emberdeep.bench`,
`python -m emberdeep.env.docgen`.

Training mode never imports pygame or moderngl: they are loaded on the first `render()` of an environment created
with `render_mode`. `import emberdeep` itself imports nothing heavy (not even the engine).

---------------------------------------------------------------------------------------------------------------

## Entry points and `EnvConfig`

| function | returns |
|---|---|
| `make_aec(scenario, **cfg)` | `EmberdeepAEC` (PettingZoo `AECEnv`) |
| `make_parallel(scenario, **cfg)` | `EmberdeepParallel` (PettingZoo `ParallelEnv`) |
| `make_gym(scenario, agent=None, others="scripted", **cfg)` | `EmberdeepGym` (Gymnasium `Env`) |
| `make_vector(scenario, num_envs, api="parallel", sync=False, seed=None, **cfg)` | `AsyncVectorEnv` (or `SyncVectorEnv`) |
| `list_scenarios()`, `scenario_info(id)` | scenario registry |
| `EnvConfig` | all options (dataclass; unknown keyword names raise `TypeError`) |

`scenario` is a registered id or a `ScenarioSpec`; `scenario_kwargs={...}` are forwarded to the scenario factory.

Option groups (see the full list below):

* **Scenario / episode** — `scenario`, `scenario_kwargs`, `seed`, `rules` (overrides of `RulesConfig`),
  `max_steps` (env steps = agent decisions), `max_rounds`, `skip_scripted`, `reaction_mode`, `upcast`.
* **Spaces** — `max_entities` (E), `n_point_targets` (P), `point_ranges`, `view_radius`, `comm_vocab`,
  `entity_order`, `enemy_hp`, `social_features`, `vision_sharing`, `allies_always_visible`, `fog_of_war_map`,
  `include_state_in_obs`, `state_entities`, `flatten_obs`, `obs_dtype`, `map_dtype`, `waiting_obs`, `illegal_action`.
* **Rewards** — `reward_weights`, `prosocial`, `svo`, `svo_group`, `mixing_matrix`, `team_reward`,
  `terminal_payoff`, `terminal_scale`, `reward_scale`.
* **Output / rendering** — `events_in_info`, `render_mode`, `render_width`, `render_height`, `render_quality`,
  `render_fps`, `render_speed`, `render_sound`, `render_pov`, `render_hud`.

<!-- BEGIN GENERATED:config -->
| field | default |
|---|---|
| `scenario` | `'goblin_ambush'` |
| `scenario_kwargs` | `{}` |
| `render_mode` | `None` |
| `seed` | `None` |
| `rules` | `{}` |
| `max_steps` | `None` |
| `max_rounds` | `None` |
| `skip_scripted` | `True` |
| `max_entities` | `16` |
| `n_point_targets` | `16` |
| `point_ranges` | `None` |
| `view_radius` | `10` |
| `comm_vocab` | `8` |
| `entity_order` | `'distance'` |
| `enemy_hp` | `'bucket'` |
| `social_features` | `True` |
| `vision_sharing` | `'none'` |
| `allies_always_visible` | `True` |
| `fog_of_war_map` | `False` |
| `upcast` | `'lowest'` |
| `reaction_mode` | `'auto'` |
| `include_state_in_obs` | `False` |
| `state_entities` | `None` |
| `flatten_obs` | `False` |
| `obs_dtype` | `'float32'` |
| `map_dtype` | `'auto'` |
| `illegal_action` | `'noop'` |
| `waiting_obs` | `'full'` |
| `reward_weights` | `None` |
| `prosocial` | `None` |
| `svo` | `None` |
| `svo_group` | `'team'` |
| `mixing_matrix` | `None` |
| `team_reward` | `None` |
| `terminal_payoff` | `None` |
| `terminal_scale` | `None` |
| `reward_scale` | `1.0` |
| `events_in_info` | `False` |
| `render_width` | `1280` |
| `render_height` | `720` |
| `render_quality` | `'medium'` |
| `render_fps` | `60` |
| `render_speed` | `1.0` |
| `render_sound` | `False` |
| `render_pov` | `'omniscient'` |
| `render_hud` | `True` |
<!-- END GENERATED:config -->

---------------------------------------------------------------------------------------------------------------

## Agents and turn semantics

Agents are the scenario's agent-controlled creatures (`agent_id`, e.g. `"party_0"`, in spawn order =
`possible_agents`). D&D is turn based, so only one creature decides at a time:

* **AEC** — `agent_selection` is the *decider*: the active creature, or the holder of a pending out-of-turn
  decision. The agent stays selected while its turn lasts (several atomic actions: move, attack, end turn...).
  Scripted creatures' turns are resolved *inside* `step()` (`skip_scripted=True`), using
  `emberdeep.ai.registry.policy_for`. Rewards accumulate for every agent between its turns
  (`_cumulative_rewards`, so `last()` returns everything that happened since the agent last acted).
  Dead and escaped agents are terminated and get their PettingZoo *dead step* (`step(None)`) immediately.
  Downed (dying) agents stay alive — they can be revived — but their turns are processed automatically by the
  engine until they are conscious again.
* **Parallel** — every step all live agents submit an action; only the decider's action is applied. Every other
  agent's mask contains only index 0 (NOOP) and `infos[a]["acting"]` is `False`. Rewards are delivered to *all*
  agents each step. `step` returns dicts for every agent that was alive before the step (including the ones that
  just terminated), then `env.agents` shrinks.
* **Gymnasium** — one agent is controlled; the wrapper advances the game (driving `others`) until the controlled
  agent has to act again. `terminated` = the agent died / escaped or the game ended, `truncated` = time limit.
  `others` is `"scripted"` (hero bots `hero_<class>`), `"random"`, a callable `(obs, info) -> action`, a policy
  object with `act(world, eid) -> Act`, or a dict agent -> any of those.
* **No stalling.** Every FREE-cost ability (offer / accept truce, rest vote, toggles...) can be used once per turn per
  ability id (unless its definition sets `data["repeatable"]`), SAY once per turn, and after
  `RulesConfig.max_actions_per_turn` (default 40, `rules={"max_actions_per_turn": n}`, 0 = off) atomic actions the
  engine ends the turn. A policy can therefore no longer freeze the game clock with free actions.
* **`reset()` never returns a finished episode.** If the scripted opening already ends the game (e.g. level-1
  heroes shot down in round 1) the next seed of the seed stream is played instead (`core.last_seed` reports it).
* `skip_scripted=False` — every engine action of a scripted creature becomes an env tick on which every agent must
  send NOOP (`infos[a]["acting"] == False`); handy for visualising the monsters' turns step by step.
* `reaction_mode="agent"` — out-of-turn choices (opportunity attacks, Shield, Uncanny Dodge, Counterspell, Divine
  Smite...) are routed to the agent as **decisions**: the decider is the reactor, its mask offers only
  `DECIDE` options, the `self` features `decision.<kind>` say what is asked and `info["decision"]` lists the option
  labels (option 0 is always "decline"). With `"auto"` (default) the engine answers them with the AI heuristics.
* `illegal_action` — a masked-out action is ignored (`"noop"`, flagged `infos[a]["invalid_action"]` and counted in
  the `invalid_action` reward component and `illegal_actions` metric), replaced by END_TURN (`"end_turn"`), or
  raises `IllegalActionError` (`"error"`). Episodes always terminate: `max_steps` counts env steps, including illegal
  ones.

`env.unwrapped.world` is the underlying `World` (read-only use recommended). `EmberdeepAEC.step` also accepts an
engine `Act` (e.g. from a scripted hero bot, as `emberdeep.play` does) instead of a flat index; it then bypasses the
masks and falls back to END_TURN if the engine rejects it.

`infos[agent]` contains: `acting` (bool), `decision` (dict or `None`), `reward_components` (dict of the
non-zero raw components since the agent's last turn for AEC / of this step for Parallel and Gym), `reward_raw`
(weighted reward before social mixing, per step), `events` (list of event dicts, if `events_in_info`),
`invalid_action` (if an illegal action was submitted), and at the end of the episode `episode` (flat metrics, see
[Metrics](#metrics)).

---------------------------------------------------------------------------------------------------------------

## Action space

`Discrete(A)` with a flat index layout and `action_mask` (int8 `[A]`):

<!-- BEGIN GENERATED:action_layout -->
| index | action |
|---|---|
| `0` | END_TURN (NOOP when you are not the decider) |
| `1..8` | MOVE N, NE, E, SE, S, SW, W, NW |
| `9`, `10` | ASCEND, DESCEND |
| `11 + k*T + t` | ABILITY `k` on target slot `t` (`k < K = 50`, `t < T = E + P = 32`) |
| `11 + K*T + e` | APPROACH entity slot `e` (`e < E = 16`) |
| `11 + K*T + E + v` | SAY token `v` (`v < V = 8`) |
| `11 + K*T + E + V + d` | DECIDE option `d` (`d < D = 8`) |

With the defaults `A = 11 + K*T + E + V + D = 1643` (`K=50`, `E=16`, `P=16`, `T=32`, `V=8`, `D=8`).
<!-- END GENERATED:action_layout -->

**Target slots.** Slots `t < E` are the *entity slots* of the observation (slot 0 = the agent itself, then
the visible creatures and objects nearest first, or by entity id with `entity_order="fixed"`; objects are capped to a
quarter of the slots while enough creatures are visible). Slots `t >= E` are *point targets*: the 8 directions x
rings (default 3 and 6 squares) of relative squares, clipped to the map and to the first sight-blocking square.
Entity-centred POINT abilities (Fireball, Misty Step) aim at the entity's square. Abilities without a target
(Dash, Dodge, Rest...) use slot 0; abilities with an area but no target (Burning Hands) use slot 0 or a point slot as
the aim.

**Masks** are computed with the engine's `ability_mask`, `target_mask`, `move_mask` and `approach_mask`:

* the factored masks `ability_mask [K]` and `target_mask [K, T]` are in the observation for autoregressive heads;
  `ability_mask[k]` is set only if ability `k` is usable *and* has at least one valid target slot;
  `action_mask[11 + k*T + t] == target_mask[k, t]`;
* END_TURN (0) is always legal for the decider; non-deciders may only send 0;
* a masked-in action always executes without error (fuzz-tested over 300 random episodes in all scenarios).

`env.action_meanings(agent)` returns an `ActionDescriptor` per legal flat action (label, ability id, icon,
cost, target entity / cell, AoE preview cells, hit chance and expected damage from `world.estimate`).

---------------------------------------------------------------------------------------------------------------

## Observation

A `gymnasium.spaces.Dict` per agent (same space for every agent). With the default configuration:

| key | shape / dtype | meaning |
|---|---|---|
| `self` | `[F_self]` float32 | the agent's own state (HP, action economy, conditions, slots, resources, ...) |
| `entities` | `[E, F_ent]` float32 | per visible entity (slot 0 = self); `entity_mask [E]` int8 says which slots are filled |
| `entity_ids` | `[E]` int32 | vocabulary id of each entity's template for embeddings (0 = empty / unknown) |
| `abilities` | `[K, F_ab]` float32 | one row per ability slot of the agent; `ability_ids [K]` int32 are vocabulary ids |
| `map` | `[21, 2R+1, 2R+1]` float32 or uint8 | egocentric map channels (R = `view_radius`, default 10) |
| `messages` | `[n_agents, V]` | last SAY token per agent (one-hot, halves with every round of age); absent if `comm_vocab == 0` |
| `action_mask` | `[A]` int8 | legal flat actions |
| `ability_mask`, `target_mask` | `[K]`, `[K, T]` int8 | factored legality |
| `state` | `[S]` | global CTDE state, only with `include_state_in_obs` |

All float features lie in `[-4, 4]`, mostly `[0, 1]`; `-1` marks "not applicable" in a few places (documented in
the tables). `obs_dtype="float16"` halves memory, `map_dtype="uint8"` stores the map channels scaled to 0..255.
With `flatten_obs=True` the observation becomes `{"observation": flat vector, "action_mask": int8 [A]}`
(concatenation of the other keys in the order `self, entities, entity_mask, entity_ids, abilities, ability_ids, map,
messages, ability_mask, target_mask, state`).

**Partial observability.**

* Creatures are listed only if the agent *sees* them (`world.sees`: line of sight, light / darkvision, invisibility,
  stealth). Hidden and invisible creatures are never listed. `vision_sharing="team"` adds what teammates see.
* Members of the agent's own faction are always listed (`allies_always_visible`, default) — a party knows where
  its members are; set it to `False` for strict line-of-sight.
* HP of non-allies follows `enemy_hp`: `"bucket"` (quarters, default), `"exact"`, `"hidden"` (not exposed). AC of
  non-allies is revealed once the observer's faction has attacked them.
* Map: `visible` / `explored` channels per agent; with `fog_of_war_map=True` the static terrain channels are
  zeroed in squares the agent has never seen.
* Social history features (`social_features`) are built from the event stream per observer: damage the entity
  dealt to me / to my allies, healing it gave, gold it gave me, betrayals it committed against my faction.

### Feature tables

These tables are generated from the code (`python -m emberdeep.env.docgen --write docs/ENV_API.md`).

#### `self`

<!-- BEGIN GENERATED:self_features -->
`self` has **129** features.

| idx | name | description |
|---|---|---|
| 0 | `hp_frac` | current HP / max HP |
| 1 | `temp_hp` | temporary HP / max HP (clipped at 2) |
| 2 | `hp_abs` | current HP / 100 |
| 3 | `max_hp` | max HP / 100 |
| 4 | `ac` | armour class / 30 |
| 5 | `speed` | walking speed in feet / 60 |
| 6 | `movement_left` | movement left this turn in feet / 60 |
| 7 | `action_avail` | 1 if the action is still unused this turn |
| 8 | `bonus_avail` | 1 if the bonus action is still unused |
| 9 | `reaction_avail` | 1 if the reaction is available |
| 10 | `free_interaction_avail` | 1 if the free object interaction is unused |
| 11 | `attacks_left` | attacks remaining in the current Attack action / 4 |
| 12 | `pos_x` | x position / map width |
| 13 | `pos_y` | y position / map height |
| 14 | `altitude` | altitude in squares / 12 |
| 15 | `cond.blinded` | condition blinded active |
| 16 | `cond.charmed` | condition charmed active |
| 17 | `cond.deafened` | condition deafened active |
| 18 | `cond.frightened` | condition frightened active |
| 19 | `cond.grappled` | condition grappled active |
| 20 | `cond.incapacitated` | condition incapacitated active |
| 21 | `cond.invisible` | condition invisible active |
| 22 | `cond.paralyzed` | condition paralyzed active |
| 23 | `cond.petrified` | condition petrified active |
| 24 | `cond.poisoned` | condition poisoned active |
| 25 | `cond.prone` | condition prone active |
| 26 | `cond.restrained` | condition restrained active |
| 27 | `cond.stunned` | condition stunned active |
| 28 | `cond.unconscious` | condition unconscious active |
| 29 | `status.dodging` | status flag dodging set |
| 30 | `status.disengaged` | status flag disengaged set |
| 31 | `status.dashed` | status flag dashed set |
| 32 | `status.hidden` | status flag hidden set |
| 33 | `status.surprised` | status flag surprised set |
| 34 | `status.dying` | status flag dying set |
| 35 | `status.stable` | status flag stable set |
| 36 | `status.dead` | status flag dead set |
| 37 | `status.concentrating` | status flag concentrating set |
| 38 | `status.raging` | status flag raging set |
| 39 | `status.readied` | status flag readied set |
| 40 | `status.fleeing` | status flag fleeing set |
| 41 | `status.surrendered` | status flag surrendered set |
| 42 | `status.wild_shaped` | status flag wild_shaped set |
| 43 | `status.helped` | status flag helped set |
| 44 | `status.flying` | status flag flying set |
| 45 | `status.escaped` | status flag escaped set |
| 46 | `status.summoned` | status flag summoned set |
| 47 | `status.mounted` | status flag mounted set |
| 48 | `status.carrying_rider` | status flag carrying_rider set |
| 49 | `status.squeezing` | status flag squeezing set |
| 50 | `status.burrowed` | status flag burrowed set |
| 51 | `exhaustion` | exhaustion level / 6 |
| 52 | `concentrating` | concentrating on a spell |
| 53 | `death_save_successes` | death-save successes / 3 |
| 54 | `death_save_failures` | death-save failures / 3 |
| 55 | `slots.1` | spell slots of level 1 remaining / 4 (clipped at 2) |
| 56 | `slots.2` | spell slots of level 2 remaining / 4 (clipped at 2) |
| 57 | `slots.3` | spell slots of level 3 remaining / 4 (clipped at 2) |
| 58 | `slots.4` | spell slots of level 4 remaining / 4 (clipped at 2) |
| 59 | `slots.5` | spell slots of level 5 remaining / 4 (clipped at 2) |
| 60 | `slots.6` | spell slots of level 6 remaining / 4 (clipped at 2) |
| 61 | `slots.7` | spell slots of level 7 remaining / 4 (clipped at 2) |
| 62 | `slots.8` | spell slots of level 8 remaining / 4 (clipped at 2) |
| 63 | `slots.9` | spell slots of level 9 remaining / 4 (clipped at 2) |
| 64 | `pact_slots` | warlock pact slots remaining / 4 |
| 65 | `pact_level` | warlock pact slot level / 5 |
| 66 | `res.action_surge` | class resource action_surge: fraction remaining, -1 if the creature does not have it |
| 67 | `res.arcane_recovery` | class resource arcane_recovery: fraction remaining, -1 if the creature does not have it |
| 68 | `res.bardic_inspiration` | class resource bardic_inspiration: fraction remaining, -1 if the creature does not have it |
| 69 | `res.channel_divinity` | class resource channel_divinity: fraction remaining, -1 if the creature does not have it |
| 70 | `res.divine_sense` | class resource divine_sense: fraction remaining, -1 if the creature does not have it |
| 71 | `res.ki` | class resource ki: fraction remaining, -1 if the creature does not have it |
| 72 | `res.lay_on_hands` | class resource lay_on_hands: fraction remaining, -1 if the creature does not have it |
| 73 | `res.rage` | class resource rage: fraction remaining, -1 if the creature does not have it |
| 74 | `res.second_wind` | class resource second_wind: fraction remaining, -1 if the creature does not have it |
| 75 | `res.sorcery_points` | class resource sorcery_points: fraction remaining, -1 if the creature does not have it |
| 76 | `res.wild_shape` | class resource wild_shape: fraction remaining, -1 if the creature does not have it |
| 77 | `res.other` | class resource other: fraction remaining, -1 if the creature does not have it |
| 78 | `gold` | log(1 + gold) / log(1 + 2000) |
| 79 | `potions` | healing potions carried / 3 |
| 80 | `class.barbarian` | player class barbarian (one-hot) |
| 81 | `class.bard` | player class bard (one-hot) |
| 82 | `class.cleric` | player class cleric (one-hot) |
| 83 | `class.druid` | player class druid (one-hot) |
| 84 | `class.fighter` | player class fighter (one-hot) |
| 85 | `class.monk` | player class monk (one-hot) |
| 86 | `class.paladin` | player class paladin (one-hot) |
| 87 | `class.ranger` | player class ranger (one-hot) |
| 88 | `class.rogue` | player class rogue (one-hot) |
| 89 | `class.sorcerer` | player class sorcerer (one-hot) |
| 90 | `class.warlock` | player class warlock (one-hot) |
| 91 | `class.wizard` | player class wizard (one-hot) |
| 92 | `is_monster` | 1 if the controlled creature is a monster (not a player class) |
| 93 | `level` | character level or CR / 10 |
| 94 | `mod.str` | ability modifier str / 5 |
| 95 | `mod.dex` | ability modifier dex / 5 |
| 96 | `mod.con` | ability modifier con / 5 |
| 97 | `mod.int` | ability modifier int / 5 |
| 98 | `mod.wis` | ability modifier wis / 5 |
| 99 | `mod.cha` | ability modifier cha / 5 |
| 100 | `mode_combat` | 1 in combat, 0 in exploration |
| 101 | `round` | round / max rounds |
| 102 | `is_decider` | 1 if this agent must act now |
| 103 | `decision.none` | pending decision kind none (one-hot, 'none' when no decision) |
| 104 | `decision.opportunity_attack` | pending decision kind opportunity_attack (one-hot, 'none' when no decision) |
| 105 | `decision.readied_trigger` | pending decision kind readied_trigger (one-hot, 'none' when no decision) |
| 106 | `decision.on_attacked` | pending decision kind on_attacked (one-hot, 'none' when no decision) |
| 107 | `decision.on_damaged` | pending decision kind on_damaged (one-hot, 'none' when no decision) |
| 108 | `decision.on_spell_cast` | pending decision kind on_spell_cast (one-hot, 'none' when no decision) |
| 109 | `decision.on_hit_rider` | pending decision kind on_hit_rider (one-hot, 'none' when no decision) |
| 110 | `decision.ally_attacked` | pending decision kind ally_attacked (one-hot, 'none' when no decision) |
| 111 | `decision.truce_offer` | pending decision kind truce_offer (one-hot, 'none' when no decision) |
| 112 | `decision.legendary` | pending decision kind legendary (one-hot, 'none' when no decision) |
| 113 | `decision.spell_choice` | pending decision kind spell_choice (one-hot, 'none' when no decision) |
| 114 | `decision.on_fall` | pending decision kind on_fall (one-hot, 'none' when no decision) |
| 115 | `decision.post_roll` | pending decision kind post_roll (one-hot, 'none' when no decision) |
| 116 | `rest_vote_self` | this agent has voted to rest |
| 117 | `rest_vote_frac` | fraction of faction members who voted to rest |
| 118 | `faction.0` | faction index 0 (one-hot) |
| 119 | `faction.1` | faction index 1 (one-hot) |
| 120 | `faction.2` | faction index 2 (one-hot) |
| 121 | `faction.3` | faction index 3 (one-hot) |
| 122 | `faction.4` | faction index 4 (one-hot) |
| 123 | `faction.5` | faction index 5 (one-hot) |
| 124 | `faction.6` | faction index 6 (one-hot) |
| 125 | `faction.7` | faction index 7 (one-hot) |
| 126 | `truce_offer_incoming` | another faction offered a truce to this agent's faction |
| 127 | `truce_offer_outgoing` | this agent's faction has an open truce offer |
| 128 | `agent_index` | (agent index + 1) / number of agents |
<!-- END GENERATED:self_features -->

#### `entities`

<!-- BEGIN GENERATED:entity_features -->
`entities[slot, :]` has **117** features with the default `comm_vocab=8` (109 without the `said.*` block; the block grows with `comm_vocab`).

| idx | name | description |
|---|---|---|
| 0 | `present` | slot is filled (same as entity_mask) |
| 1 | `kind.creature` | entity kind creature |
| 2 | `kind.door` | entity kind door |
| 3 | `kind.chest` | entity kind chest |
| 4 | `kind.loot` | entity kind loot |
| 5 | `kind.lever` | entity kind lever |
| 6 | `kind.trap` | entity kind trap |
| 7 | `kind.altar` | entity kind altar |
| 8 | `kind.exit` | entity kind exit |
| 9 | `cond.blinded` | condition blinded active |
| 10 | `cond.charmed` | condition charmed active |
| 11 | `cond.deafened` | condition deafened active |
| 12 | `cond.frightened` | condition frightened active |
| 13 | `cond.grappled` | condition grappled active |
| 14 | `cond.incapacitated` | condition incapacitated active |
| 15 | `cond.invisible` | condition invisible active |
| 16 | `cond.paralyzed` | condition paralyzed active |
| 17 | `cond.petrified` | condition petrified active |
| 18 | `cond.poisoned` | condition poisoned active |
| 19 | `cond.prone` | condition prone active |
| 20 | `cond.restrained` | condition restrained active |
| 21 | `cond.stunned` | condition stunned active |
| 22 | `cond.unconscious` | condition unconscious active |
| 23 | `status.dodging` | status flag dodging (hidden bit is masked for non-allies) |
| 24 | `status.disengaged` | status flag disengaged (hidden bit is masked for non-allies) |
| 25 | `status.dashed` | status flag dashed (hidden bit is masked for non-allies) |
| 26 | `status.hidden` | status flag hidden (hidden bit is masked for non-allies) |
| 27 | `status.surprised` | status flag surprised (hidden bit is masked for non-allies) |
| 28 | `status.dying` | status flag dying (hidden bit is masked for non-allies) |
| 29 | `status.stable` | status flag stable (hidden bit is masked for non-allies) |
| 30 | `status.dead` | status flag dead (hidden bit is masked for non-allies) |
| 31 | `status.concentrating` | status flag concentrating (hidden bit is masked for non-allies) |
| 32 | `status.raging` | status flag raging (hidden bit is masked for non-allies) |
| 33 | `status.readied` | status flag readied (hidden bit is masked for non-allies) |
| 34 | `status.fleeing` | status flag fleeing (hidden bit is masked for non-allies) |
| 35 | `status.surrendered` | status flag surrendered (hidden bit is masked for non-allies) |
| 36 | `status.wild_shaped` | status flag wild_shaped (hidden bit is masked for non-allies) |
| 37 | `status.helped` | status flag helped (hidden bit is masked for non-allies) |
| 38 | `status.flying` | status flag flying (hidden bit is masked for non-allies) |
| 39 | `status.escaped` | status flag escaped (hidden bit is masked for non-allies) |
| 40 | `status.summoned` | status flag summoned (hidden bit is masked for non-allies) |
| 41 | `status.mounted` | status flag mounted (hidden bit is masked for non-allies) |
| 42 | `status.carrying_rider` | status flag carrying_rider (hidden bit is masked for non-allies) |
| 43 | `status.squeezing` | status flag squeezing (hidden bit is masked for non-allies) |
| 44 | `status.burrowed` | status flag burrowed (hidden bit is masked for non-allies) |
| 45 | `size` | size category / 5 |
| 46 | `ctype.aberration` | creature type aberration |
| 47 | `ctype.beast` | creature type beast |
| 48 | `ctype.celestial` | creature type celestial |
| 49 | `ctype.construct` | creature type construct |
| 50 | `ctype.dragon` | creature type dragon |
| 51 | `ctype.elemental` | creature type elemental |
| 52 | `ctype.fey` | creature type fey |
| 53 | `ctype.fiend` | creature type fiend |
| 54 | `ctype.giant` | creature type giant |
| 55 | `ctype.humanoid` | creature type humanoid |
| 56 | `ctype.monstrosity` | creature type monstrosity |
| 57 | `ctype.ooze` | creature type ooze |
| 58 | `ctype.plant` | creature type plant |
| 59 | `ctype.undead` | creature type undead |
| 60 | `level` | character level or CR / 10 |
| 61 | `is_agent` | controlled through the RL API |
| 62 | `agent_index` | (agent index + 1) / number of agents (0 if scripted) |
| 63 | `is_active` | its turn is in progress |
| 64 | `initiative` | position in the turn order / length (-1 outside combat) |
| 65 | `concentrating` | concentrating on a spell |
| 66 | `downed` | at 0 HP (dying or stable) |
| 67 | `surrendered` | has surrendered |
| 68 | `fleeing` | is fleeing |
| 69 | `faction.0` | faction index 0 (one-hot) |
| 70 | `faction.1` | faction index 1 (one-hot) |
| 71 | `faction.2` | faction index 2 (one-hot) |
| 72 | `faction.3` | faction index 3 (one-hot) |
| 73 | `faction.4` | faction index 4 (one-hot) |
| 74 | `faction.5` | faction index 5 (one-hot) |
| 75 | `faction.6` | faction index 6 (one-hot) |
| 76 | `faction.7` | faction index 7 (one-hot) |
| 77 | `rel.self` | relation to the observer: self |
| 78 | `rel.ally` | relation to the observer: ally |
| 79 | `rel.neutral` | relation to the observer: neutral |
| 80 | `rel.truce` | relation to the observer: truce |
| 81 | `rel.hostile` | relation to the observer: hostile |
| 82 | `dx` | x offset to the observer / 20 (clipped) |
| 83 | `dy` | y offset to the observer / 20 (clipped) |
| 84 | `dz` | altitude offset / 6 |
| 85 | `dist` | edge-to-edge distance in squares / 20 (clipped at 2) |
| 86 | `in_reach` | within the observer's melee reach |
| 87 | `in_range` | within range of the observer's longest-range attack ability |
| 88 | `has_los` | the observer sees this entity (line of sight, light, stealth) |
| 89 | `cover` | cover from the observer / 3 (0 none .. 1 total) |
| 90 | `hp_frac` | HP fraction (allies exact; others per `enemy_hp`: bucket / exact / hidden=0) |
| 91 | `hp_known` | 1 if HP information is exposed for this entity |
| 92 | `hp_abs` | HP / 100 (allies and `enemy_hp='exact'` only) |
| 93 | `ac` | armour class / 30 (allies; others once the observer's faction has attacked them), else 0 |
| 94 | `ac_known` | 1 if `ac` is valid |
| 95 | `truce_offer_in` | its faction offered a truce to the observer's faction |
| 96 | `truce_offer_out` | observer's faction offered a truce to its faction |
| 97 | `said.0` | last SAY token 0 (one-hot, decays with age) |
| 98 | `said.1` | last SAY token 1 (one-hot, decays with age) |
| 99 | `said.2` | last SAY token 2 (one-hot, decays with age) |
| 100 | `said.3` | last SAY token 3 (one-hot, decays with age) |
| 101 | `said.4` | last SAY token 4 (one-hot, decays with age) |
| 102 | `said.5` | last SAY token 5 (one-hot, decays with age) |
| 103 | `said.6` | last SAY token 6 (one-hot, decays with age) |
| 104 | `said.7` | last SAY token 7 (one-hot, decays with age) |
| 105 | `dmg_to_me` | social: damage it dealt to the observer / 50 (needs social_features) |
| 106 | `dmg_to_allies` | social: damage it dealt to the observer's allies / 50 |
| 107 | `heal_to_me` | social: healing it gave the observer / 50 |
| 108 | `heal_to_allies` | social: healing it gave the observer's allies / 50 |
| 109 | `gold_to_me` | social: gold it gave the observer / 100 |
| 110 | `betrayals` | social: betrayals (attacks on allies / truce partners) by it against the observer's faction / 3 |
| 111 | `obj_open` | object: door open / chest opened / lever on |
| 112 | `obj_closed` | object: door or chest closed |
| 113 | `obj_locked` | object: locked |
| 114 | `obj_broken` | object: door broken / trap disarmed or triggered |
| 115 | `obj_value` | object: log(1 + gold value) / log(1 + 1000) of loot or chest contents |
| 116 | `obj_charges` | object: altar charges / capacity |
<!-- END GENERATED:entity_features -->

#### `abilities`

<!-- BEGIN GENERATED:ability_features -->
`abilities[k, :]` has **92** features.

| idx | name | description |
|---|---|---|
| 0 | `exists` | the slot holds an ability |
| 1 | `available` | usable now with at least one valid target |
| 2 | `kind.basic` | ability kind basic |
| 3 | `kind.weapon` | ability kind weapon |
| 4 | `kind.spell` | ability kind spell |
| 5 | `kind.feature` | ability kind feature |
| 6 | `kind.item` | ability kind item |
| 7 | `kind.monster` | ability kind monster |
| 8 | `cost.action` | action cost action |
| 9 | `cost.bonus_action` | action cost bonus_action |
| 10 | `cost.reaction` | action cost reaction |
| 11 | `cost.free` | action cost free |
| 12 | `cost.object_interaction` | action cost object_interaction |
| 13 | `cost.attack` | action cost attack |
| 14 | `cost.movement` | action cost movement |
| 15 | `cost.legendary` | action cost legendary |
| 16 | `cost.special` | action cost special |
| 17 | `target.none` | target type none |
| 18 | `target.self` | target type self |
| 19 | `target.creature` | target type creature |
| 20 | `target.ally` | target type ally |
| 21 | `target.enemy` | target type enemy |
| 22 | `target.self_or_ally` | target type self_or_ally |
| 23 | `target.point` | target type point |
| 24 | `target.object` | target type object |
| 25 | `target.downed_ally` | target type downed_ally |
| 26 | `target.any_creature` | target type any_creature |
| 27 | `target.creature_or_object` | target type creature_or_object |
| 28 | `range` | range in squares / 30 |
| 29 | `long_range` | long range in squares / 30 |
| 30 | `reach` | reach in squares / 6 |
| 31 | `area.none` | area shape none |
| 32 | `area.sphere` | area shape sphere |
| 33 | `area.cube` | area shape cube |
| 34 | `area.cone` | area shape cone |
| 35 | `area.line` | area shape line |
| 36 | `area.cylinder` | area shape cylinder |
| 37 | `area.emanation` | area shape emanation |
| 38 | `area_size` | area size in squares / 12 |
| 39 | `avg_damage` | average damage / 30 (0 if none) |
| 40 | `dtype.acid` | damage type acid |
| 41 | `dtype.bludgeoning` | damage type bludgeoning |
| 42 | `dtype.cold` | damage type cold |
| 43 | `dtype.fire` | damage type fire |
| 44 | `dtype.force` | damage type force |
| 45 | `dtype.lightning` | damage type lightning |
| 46 | `dtype.necrotic` | damage type necrotic |
| 47 | `dtype.piercing` | damage type piercing |
| 48 | `dtype.poison` | damage type poison |
| 49 | `dtype.psychic` | damage type psychic |
| 50 | `dtype.radiant` | damage type radiant |
| 51 | `dtype.slashing` | damage type slashing |
| 52 | `dtype.thunder` | damage type thunder |
| 53 | `avg_healing` | average healing / 20 |
| 54 | `spell_level` | spell level / 9 (-1 for non-spells) |
| 55 | `uses_left` | slots or uses remaining / 5 (-1 if unlimited) |
| 56 | `concentration` | requires concentration |
| 57 | `save.str` | saving throw ability str |
| 58 | `save.dex` | saving throw ability dex |
| 59 | `save.con` | saving throw ability con |
| 60 | `save.int` | saving throw ability int |
| 61 | `save.wis` | saving throw ability wis |
| 62 | `save.cha` | saving throw ability cha |
| 63 | `attack_bonus` | attack bonus / 10 (0 if none) |
| 64 | `save_dc` | save DC / 20 (0 if none) |
| 65 | `tag.attack` | tag attack |
| 66 | `tag.melee` | tag melee |
| 67 | `tag.ranged` | tag ranged |
| 68 | `tag.damage` | tag damage |
| 69 | `tag.heal` | tag heal |
| 70 | `tag.healing` | tag healing |
| 71 | `tag.buff` | tag buff |
| 72 | `tag.debuff` | tag debuff |
| 73 | `tag.control` | tag control |
| 74 | `tag.aoe` | tag aoe |
| 75 | `tag.support` | tag support |
| 76 | `tag.movement` | tag movement |
| 77 | `tag.stealth` | tag stealth |
| 78 | `tag.social` | tag social |
| 79 | `tag.defense` | tag defense |
| 80 | `tag.item` | tag item |
| 81 | `tag.object` | tag object |
| 82 | `tag.utility` | tag utility |
| 83 | `tag.reaction` | tag reaction |
| 84 | `tag.thrown` | tag thrown |
| 85 | `tag.teleport` | tag teleport |
| 86 | `tag.summon` | tag summon |
| 87 | `tag.light` | tag light |
| 88 | `tag.exploration` | tag exploration |
| 89 | `tag.perception` | tag perception |
| 90 | `tag.rest` | tag rest |
| 91 | `tag.ritual` | tag ritual |
<!-- END GENERATED:ability_features -->

#### `map` channels

<!-- BEGIN GENERATED:map_channels -->
| channel | name | description |
|---|---|---|
| 0 | `in_bounds` | square lies inside the map |
| 1 | `opaque` | wall, pillar or other sight-blocking terrain / closed door |
| 2 | `door_closed` | closed or locked door |
| 3 | `door_open` | open or broken door |
| 4 | `difficult` | difficult terrain (double movement) |
| 5 | `hazard` | lava or other damaging terrain |
| 6 | `water` | shallow or deep water |
| 7 | `pit` | pit / chasm |
| 8 | `cover` | low wall or statue (half / three-quarter cover) |
| 9 | `visible` | currently visible to the observer (or its team with vision sharing) |
| 10 | `explored` | seen at some point this episode |
| 11 | `light` | light level at the square (0 dark, 0.5 dim, 1 bright) |
| 12 | `allies` | square occupied by an ally (not self) |
| 13 | `enemies` | square occupied by a hostile creature visible to the observer |
| 14 | `neutrals` | square occupied by a neutral or truce creature |
| 15 | `objects` | chest, loot, lever, altar, detected trap or exit marker |
| 16 | `zone_harmful` | inside a harmful zone (fire, web, stinking cloud...) |
| 17 | `zone_helpful` | inside a beneficial or neutral zone (light, silence...) |
| 18 | `reachable` | reachable with the movement left this turn (decider only) |
| 19 | `threat` | inside the melee reach of a visible hostile creature |
| 20 | `objective` | exit square or region named by an objective of the observer's faction |
<!-- END GENERATED:map_channels -->

---------------------------------------------------------------------------------------------------------------

## Global state (CTDE)

`env.state()` (AEC, Parallel, Gym, vector `venv.state()`) returns a float vector described by `env.state_space`:
all creatures in a fixed order with full (unobscured) features, a downsampled full-map summary and scenario
progress. With `include_state_in_obs=True` the same vector is `obs["state"]` for every agent.

<!-- BEGIN GENERATED:state_layout -->
Per creature slot (20 fields; slots are the agents in `possible_agents` order followed by every other creature by entity id; dead creatures keep their slot):

| field | name | description |
|---|---|---|
| 0 | `present` | creature slot is filled and present on the map |
| 1 | `hp_frac` | HP / max HP |
| 2 | `hp_abs` | HP / 100 |
| 3 | `ac` | AC / 30 |
| 4 | `pos_x` | x / map width |
| 5 | `pos_y` | y / map height |
| 6 | `z` | altitude / 12 |
| 7 | `size` | size / 5 |
| 8 | `is_agent` | controlled through the API |
| 9 | `agent_index` | (agent index + 1) / number of agents |
| 10 | `is_active` | its turn is in progress |
| 11 | `downed` | at 0 HP |
| 12 | `dead` | dead |
| 13 | `escaped` | escaped through an exit |
| 14 | `concentrating` | concentrating |
| 15 | `gold` | log(1 + gold) / log(1 + 2000) |
| 16 | `wealth` | log(1 + wealth incl. items) / log(1 + 2000) |
| 17 | `level` | level or CR / 10 |
| 18 | `slots` | total spell slots remaining / 12 |
| 19 | `movement_left` | movement left / 60 |

Map summary: 7 channels downsampled (block average) to 16x16:

| channel | name |
|---|---|
| 0 | `blocked` |
| 1 | `hazard` |
| 2 | `agents` |
| 3 | `hostiles` |
| 4 | `objects` |
| 5 | `objective` |
| 6 | `explored_by_team` |

Scenario block (92 values): round fraction, combat flag, env-step fraction, fraction of agents alive, (done, failed) for up to 8 objectives, the 8x8 faction relation matrix ((relation+1)/4), and the log-scaled gold of each agent.
<!-- END GENERATED:state_layout -->

---------------------------------------------------------------------------------------------------------------

## Rewards

Per-agent **components** are computed from the engine's event stream every step and exposed in
`infos[a]["reward_components"]`:

```
events --RewardEngine.process--> components C[i, c]           (non-negative magnitudes: HP, gp, counts)
       --weights (defaults < scenario RewardSpec < EnvConfig.reward_weights)--> raw reward r_i
       --social mixing  r' = M r   (M from prosocial / svo / team_reward / mixing_matrix)-->
       --reward_scale--> reward
```

Weights carry the sign (`damage_taken` has a negative weight). Defaults are the shared ones of
`docs/SCENARIO_DESIGN.md` (`rewards.DEFAULT_WEIGHTS`); a scenario's `RewardSpec.weights` override them and
`EnvConfig.reward_weights` overrides both. Unknown component names raise `KeyError`.

<!-- BEGIN GENERATED:reward_components -->
| component | default weight | definition |
|---|---|---|
| `damage_dealt` | 0.004 | HP of damage dealt to hostile creatures (credit follows the damage source; summons credit their summoner) |
| `damage_taken` | -0.004 | HP of damage suffered (any source) |
| `friendly_damage` | -0.01 | HP of damage dealt to creatures of the same faction or an allied one |
| `kills` | 0.0 | hostile creatures killed (final blow) |
| `kill_xp` | 0.0005 | XP value of hostile creatures killed (final blow) |
| `team_xp` | 0.0 | XP value of hostile creatures killed by anyone in the agent's faction |
| `healing_given` | 0.004 | HP restored to non-hostile creatures other than self |
| `healing_received` | 0.0 | HP restored to this agent by any source (including itself) |
| `revives` | 0.2 | allies brought back from 0 HP by this agent |
| `downed` | -0.1 | times this agent dropped to 0 HP |
| `death` | -0.5 | 1 when this agent dies |
| `gold` | 0.001 | signed change of personal gold (pickups, gifts, donations, spell material costs) |
| `item_value` | 0.0 | signed change of the gp value of carried items (only tracked when its weight is non-zero) |
| `gold_given` | 0.0 | gp handed to other creatures (give_gold) |
| `gold_received` | 0.0 | gp received from other creatures |
| `objective` | 1.0 | objectives completed by the agent's faction (ObjectiveSpec.reward, else 1 for non-final objectives) |
| `victory` | 1.0 | 1 at the end of an episode that the agent's faction won |
| `defeat` | -1.0 | 1 at the end of an episode that ended (not truncated) without the agent's faction winning |
| `escaped` | 0.0 | 1 when the agent escapes through an exit |
| `survived` | 0.0 | 1 at episode end if the agent is alive or escaped |
| `time` | -0.0005 | game rounds elapsed while the agent was in play |
| `exploration` | 0.0 | newly explored map squares (only tracked when its weight is non-zero) |
| `betrayal` | 0.0 | attacks on allied / truce partners initiated by the agent |
| `truce` | 0.0 | truces formed (credited to every agent of both factions) |
| `spared` | 0.0 | surrendered hostile creatures still alive at episode end that this agent never attacked after they yielded |
| `damage_neutral` | 0.0 | HP of damage dealt to neutral or truce-bound creatures |
| `damage_surrendered` | 0.0 | HP of damage dealt to creatures that had surrendered |
| `terminal` | 1.0 | wealth (gold + item value) x terminal_scale, paid at episode end per the `terminal_payoff` mode (none \| gold_if_alive \| gold_if_escaped \| gold); scenario extras from `runtime.terminal_payoffs` (escort fee, prize, survival bonus...) are added as extra components at weight 1 |
| `invalid_action` | 0.0 | illegal (masked-out) actions submitted |
<!-- END GENERATED:reward_components -->

Credit rules: damage / kills / healing are credited to the agent that caused them (summons credit their
summoner); relations are evaluated *at the time of the event*, so a hit on a truce partner counts as
`damage_neutral` / `friendly_damage` even though the betrayal breaks the truce a moment later. `victory` /
`defeat` are paid when an episode ends (not when it is truncated); `survived` also on truncation.

**Social mixing** (precedence: `mixing_matrix` > `team_reward` > `svo` > `prosocial`; scenario `RewardSpec`
provides defaults for `prosocial` and `team_reward`). "Team" = agents of the same faction.

| option | formula |
|---|---|
| `prosocial=a` (float or `{agent: a}`) | `r'_i = (1-a) r_i + a * mean_{j != i, same team} r_j` |
| `svo=theta` degrees (float or dict) | `r'_i = cos(theta) r_i + sin(theta) * mean_{j != i} r_j` (`svo_group="team"` or `"all"`) |
| `team_reward=True` | `r'_i = mean_{j in team(i)} r_j` |
| `mixing_matrix=W` | `r' = W r`, agents in `possible_agents` order |

An agent alone in its team is unaffected by `prosocial` / `svo` (with `svo_group="team"`).

**Terminal payoffs.** `terminal_payoff` (scenario `RewardSpec`, overridable) pays at the end of an episode
(terminated or truncated) `wealth x terminal_scale` as the `terminal` component: `"gold_if_alive"`,
`"gold_if_escaped"`, `"gold"` (wealth = gold + value of carried items). Scenario-specific extras from
`runtime.terminal_payoffs` (escort fee, prize, survival bonus...) are added with weight 1 and listed under their
own names in `reward_components`.

---------------------------------------------------------------------------------------------------------------

## Metrics

`env.episode_metrics()` returns `{"agents": {agent: {...}}, "global": {...}, "scenario": {...}}` at any time;
`infos[a]["episode"]` (flat dict: the agent's values, the global ones and `scenario/<key>`) appears when the
episode ends. `scenario` comes from `runtime.metrics(world)` (truces, betrayals, common-pool depletion time,
donations, boss engagement, bribes, ...; see `docs/SCENARIOS.md`).

<!-- BEGIN GENERATED:metrics -->
| metric | definition |
|---|---|
| `return` | episode return of the agent (mixed reward, after social mixing and reward_scale) |
| `return_raw` | episode return before social mixing |
| `survived` | 1 if the agent is alive or escaped |
| `escaped` | 1 if the agent left through an exit |
| `win` | 1 if the agent's faction won (episode ended and the faction is in `outcome['winners']`) |
| `loss` | 1 if the episode ended without the agent's faction winning (truncation counts as neither) |
| `damage_dealt` | HP of damage dealt to hostile creatures |
| `damage_taken` | HP of damage taken |
| `friendly_damage` | HP of damage dealt to allies |
| `healing_given` | HP healed on others |
| `revives_given` | allies revived from 0 HP |
| `gold_given` | gp given away |
| `gold_received` | gp received |
| `wealth` | final gold + item value |
| `kills` | hostile creatures killed |
| `betrayals` | attacks on allies / truce partners |
| `free_riding` | loot share minus damage-taken share within the agent's team (positive = takes more than it risks) |
| `team_return` | mean mixed return of the agent's team (key `team_return/<faction>` globally) |
| `collective_return` | sum of the mixed returns of all agents (efficiency) |
| `equality` | 1 - Gini of the (non-negative part of the) agent returns; 1 = perfectly equal |
| `return_std` | standard deviation of agent returns |
| `wealth_gini` | Gini coefficient of final agent wealth |
| `survival_rate` | fraction of agents alive or escaped at the end |
| `win_rate` | fraction of agents whose faction won |
| `rounds` | game rounds played |
| `steps` | env steps (agent decisions) taken |
| `heals_given` | total HP healed on others by agents |
| `revives` | total revives by agents |
| `gold_given_total` | total gp given by agents |
| `betrayals_total` | total betrayals by agents |
| `truces_formed` | truces formed between factions |
| `truces_broken` | truces that returned to hostility |
| `friendly_fire` | total HP of damage between allied creatures |
| `free_riding_index` | largest per-agent free-riding value |
| `illegal_actions` | masked-out actions submitted |
<!-- END GENERATED:metrics -->

---------------------------------------------------------------------------------------------------------------

## Vector environments

`make_vector(scenario, num_envs, api="parallel")` creates `num_envs` copies in worker subprocesses (`spawn`
context). Observations are written by the workers straight into `multiprocessing.shared_memory` buffers; only a
tiny message per step travels over a pipe. `sync=True` (or `SyncVectorEnv`) runs everything in-process.

```python
obs, infos = venv.reset(seed=0)           # obs: dict key -> ndarray [num_envs, n_agents, ...]
obs, rewards, terminations, truncations, infos = venv.step(actions)   # actions int [num_envs, n_agents]
```

* `rewards float32`, `terminations bool`, `truncations bool` have shape `[num_envs, n_agents]`; a terminated
  (dead / escaped) agent stays `terminated` and receives 0 reward until the episode ends.
* Only the decider's action of each sub-environment is applied; mask rows of the others allow only NOOP.
* `infos` is a list of dicts: `acting` and `alive` (lists over agents), `seed`; at the end of an episode also
  `episode_done`, `episode` (`{agent: flat metrics}`) and `final_observation` (arrays `[n_agents, ...]` of the
  last state). **Environments reset automatically**: the returned `obs` of a finished environment already belong
  to the next episode, whose seed comes from the environment's own deterministic seed stream.
* `venv.state()` (`[num_envs, S]`), `venv.episode_metrics()`, `venv.call(name, *args)` (any `EmberdeepCore`
  method in every worker), `venv.close()` (also on exit; shared memory is unlinked).
* `seed=` of the constructor / `reset(seed=s)` gives environment `i` the seed `s + i`.
* `waiting_obs="stale"` roughly doubles vector throughput: only the acting agent's observation is recomputed.

---------------------------------------------------------------------------------------------------------------

## Rendering, replays, human play

* `render_mode="human"` — `env.render()` pushes the frames and events produced since the last call to
  `emberdeep.render.Renderer` and blocks until the animations finish (`play_until_idle`). `render_mode="rgb_array"`
  returns `renderer.capture()` (offscreen `H x W x 3 uint8`). Options: `render_width/height/quality/fps/speed/sound/pov/hud`.
  `env.set_annotations({agent: {"value": v, "top_actions": [(label, p), ...], "note": "..."}})` feeds the research
  overlay.
* `emberdeep.env.render_adapter.FrameBuilder` turns a world into the renderer contract (`StaticScene` once per
  episode, `FrameState` per step: entities with conditions / effects / resources / inventory / reward components,
  zones, visibility per agent and faction, doors, SAY messages, relations, objectives, pending decision, metrics).
* `emberdeep.replay.Recorder(env, "game.edr")` wraps an AEC, Parallel or Gym environment and writes a gzip-pickled
  `StaticScene` plus `[(FrameState, events)]` (one file per episode; later ones get `_<n>` suffixes);
  `load_replay(path)` reads it. Replays are pickles: load only files you trust.
* `python -m emberdeep.viewer game.edr` plays a replay in the renderer: Space / HUD buttons pause and speed, **Right /
  Left** step one frame, Home / End / PageUp / PageDown jump, `Tab` cycles the POV, `F12` screenshots;
  `--export-frames DIR [--sim-fps N] [--every K]` renders PNGs offscreen; `--info` prints the metadata.
* `python -m emberdeep.play --scenario goblin_ambush [--control party_0[,party_1]|all] [--others scripted|random]
  [--record out.edr] [--watch]` is human play: controlled agents act through the renderer's action bar
  (`renderer.request_human_action(agent_id, env.action_meanings(agent))`), `--reaction-mode agent` (default) lets you
  answer reactions yourself, every other agent is a hero bot or random; `--watch` spectates an all-scripted game.

---------------------------------------------------------------------------------------------------------------

## Determinism, cloning, seeding

* `reset(seed=s)` always uses `s` verbatim: the same seed and the same actions give bit-identical observations,
  rewards and `world.state_hash()` (`tests/test_env_determinism.py`). `reset()` without a seed draws the next
  episode seed from a stream derived from `EnvConfig.seed` (entropy if `None`).
* `env.clone_state()` / `env.restore_state(snapshot)` (AEC, Parallel, Gym) snapshot the world (`world.clone()`),
  reward / explored-map / social-history bookkeeping and the seed stream. Snapshots are reusable and independent
  of the live environment, so they serve planning and MCTS: snapshot, explore a branch, restore, explore another.
  The scenario runtime keeps its state in `world.stats`, so it is cloned with the world.
* Scripted policies keep no hidden state and take their tie-breaking randomness from a hash of the world state, so
  clones replay identically.

---------------------------------------------------------------------------------------------------------------

## Performance

Measured on this repository's CI box (4 vCPUs that deliver about 2 cores of throughput, shared with other jobs, so numbers
vary by +-15%), `goblin_ambush` (4 PCs vs 6 goblins + a bugbear), random legal agents, scripted monsters and a full
observation (+ masks) for the next agent included, Python 3.13 (`python -m emberdeep.bench --scenario goblin_ambush --envs 4`):

| benchmark | agent steps / s |
|---|---|
| AEC (one core) | ~1,000 |
| Parallel (fresh obs for 4 agents each step) | ~700 |
| Parallel, `waiting_obs="stale"` | ~1,000 |
| 4 sub-environments in-process (`SyncVectorEnv`), full / stale | ~610 / ~900 |
| 4 worker processes (`AsyncVectorEnv`), full / stale | ~960 / ~1,300 |

(Before perf2: AEC ~790, Parallel ~580, async-vec ~780 on the same box.)

Where the time goes (AEC, per agent step, about 1.0 ms): observation 0.17 ms, legality masks and slots 0.38 ms
(approach mask 0.10 ms -- one Dijkstra field per decision, the largest single item; target masks 0.07 ms, ability
mask 0.05 ms, candidate slots 0.05 ms, move mask 0.02 ms), the agent's engine step 0.17 ms and the scripted monsters'
AI 0.10 ms, bookkeeping the rest.  perf2 made the masks cheaper by economy-gating `usable()` (spent action / bonus /
movement classes are skipped), distance / kind / relation pre-filters for `valid_target`, a memoised, Numba-backed
`move_mask`, a Numba field-of-view window and point-slot kernels, and scalar `_step_cost` for size-1 walkers; the
reference implementations (`ability_mask_ref`, `target_mask_ref`, `move_mask_ref`, `visible_cells_ref`,
`point_slots_ref_into`) stay in the code and `tests/test_perf2_masks.py` / `tests/test_perf_*.py` check the fast
paths against them bit for bit on every scenario.  `World.estimate` / `estimate_many` never consume `world.rng`
(random defences such as Sanctuary and Mirror Image contribute expected values).  `python benchmarks/bench_perf2.py
--profile` prints the throughput and a cProfile summary.  The DESIGN target of 1,500 agent steps/s per core is **not**
reached on this hardware (the engine/AI turns and the approach Dijkstra are the floor); async vector envs with
`waiting_obs="stale"` do exceed it.  Practical advice:

* use `waiting_obs="stale"` for Parallel / vector training when only the acting agent's observation matters;
* shrink what you do not need: `max_entities`, `view_radius`, `n_point_targets`, `obs_dtype="float16"`,
  `map_dtype="uint8"`, `social_features=False`;
* scale with processes (`make_vector(..., num_envs=cores)`): sub-environments do not share state, so throughput
  scales with physical cores.

---------------------------------------------------------------------------------------------------------------

## Testing and known limitations

`pytest -q` runs `tests/test_env_api.py` (PettingZoo `api_test` / `parallel_api_test`, Gymnasium `check_env`,
observation containment for every scenario), `test_env_masks.py` (layout, slots, 300-episode legality fuzz, reaction
decisions, SAY, visibility), `test_env_rewards.py`, `test_env_determinism.py` (including clone / restore),
`test_env_vector.py` and `test_render_adapter.py` (scene / frame validity for every scenario, rgb_array render,
replay round trip; the GL tests skip themselves without a usable display, use `DISPLAY=:99`).

* The APPROACH mask includes a workaround for an engine path-planning inconsistency (corner cutting), see
  `docs/dev/contract_requests/env.md`.
* `skip_scripted=False` yields one env tick per scripted engine action; in AEC mode the idle agents are selected in
  round-robin order and must send NOOP.
* The metric `equality` uses the non-negative part of the agents' returns (as in Melting Pot); `return_std` is also
  reported for signed returns.
