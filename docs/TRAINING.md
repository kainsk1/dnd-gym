# Training agents on Emberdeep (reference MAPPO / IPPO)

`examples/train_mappo.py` is the reference starting point for learning in Emberdeep: PPO with parameter sharing,
a centralised (MAPPO) or decentralised (IPPO) critic, a network that uses the structure of the observation and of
the flat action space, turn-based credit assignment, social-dilemma knobs on the command line, CSV logging and
checkpoints. It runs on a laptop CPU (PyTorch, no GPU needed).

* [Quick start](#quick-start) - [Architecture](#architecture) - [Turn-based credit assignment](#turn-based-credit-assignment)
* [Free actions and the turn limit](#free-actions-and-the-turn-limit) - [Hyper-parameters](#hyper-parameters)
* [Social-dilemma experiments](#social-dilemma-experiments) - [Logged metrics](#logged-metrics)
* [Expected results and runtimes](#expected-results-and-runtimes) - [Watching a trained agent](#watching-a-trained-agent)
* [Limitations](#limitations)

---------------------------------------------------------------------------------------------------------------

## Quick start

```bash
python examples/train_mappo.py --scenario goblin_ambush_small --algo mappo --envs 8 \
       --env-kw view_radius=6 --time-limit 8 --name goblins        # ~8 min on 4 shared CPU cores
python examples/evaluate.py runs/goblins/final.pt --episodes 100   # win rate vs the scripted goblins
python examples/evaluate.py runs/goblins/final.pt --episodes 2 --record goblins.edr --annotate
python -m emberdeep.viewer goblins.edr                             # policy-insight HUD: value + top-5 actions
```

Files: `examples/models.py` (networks), `examples/rollout.py` (trajectory collector, GAE, turn limiter),
`examples/train_mappo.py` (CLI + PPO loop), `examples/evaluate.py`. Tests: `tests/test_train_example.py`.

## Architecture

```
 observation (Dict)                                          shared encoders (per agent)
 ------------------                                          ---------------------------
 map  [21,R,R] (uint8)  ---------- conv3x3/2 - conv3x3 - conv3x3/2 - flatten - Linear ---> 128 --+
 self [122]             ---------- MLP --------------------------------------------------> 64  --+
 entities [E,113]  \                                                                              |
 entity_ids [E]     >-- Linear + Embedding -> LayerNorm -> 2-layer Transformer ------------+      |
 entity_mask [E]   /     (key padding mask; slot 0 = self token (+ self MLP) = query / CLS)  |      |
                         token 0 --------------------------------------------------------> 64  --+--> concat
                         tokens 0..E-1 = T_e (target embeddings) -----------------------+   |      |      |
 abilities [K,92]  \                                                                     |   |      |      v
 ability_ids [K]    >-- Linear + Embedding -> MLP -> A_k [K,64] --- masked mean -----> 64  --+      |   trunk MLP
 messages [n,V]         ---------- MLP ---------------------------------------------> 32  --+      |   256-256
 agent index            ---------- Embedding ---------------------------------------> 16  --+      |      |
                                                                                                  |      h
 policy heads (flat Discrete(A), layout of docs/ENV_API.md)                                       |      |
 -----------------------------------------------------------------------------------              |      |
 END, MOVE x8, ASCEND, DESCEND   [11]   Linear(h)                                                 |      |
 ABILITY k on target t  [K*T]           <q_k, W T_t>/sqrt(d) + b_k      q_k = Wq A_k + Wh h <------+------+
                                        T_t = entity token t (t < E) | learned point embedding + Wp h (t >= E)
 APPROACH e             [E]             MLP([T_e, Wa h])
 SAY v [V], DECIDE d [D]                Linear(h)
        |
        +-- concatenated at 11 + k*T + t, 11 + K*T + e, ...  == ActionLayout   ->  logits.masked_fill(~action_mask, -1e9)
                                                                                 ->  Categorical

 value:  IPPO   V(h)                          decentralised critic (own observation, shared trunk)
         MAPPO  MLP([env.state(), onehot(agent)])   centralised critic (CTDE; obs["state"] via include_state_in_obs)
```

About 0.75 M parameters (actor + IPPO critic) / 1.34 M with the MAPPO critic (the state has 2,364 inputs on
`goblin_ambush_small`). `assemble_flat_logits` in `models.py` is unit-tested against `ActionLayout.decode`.
Everything is derived from the environment's spaces (`ModelSpec.from_env`), so other `max_entities`,
`n_point_targets`, `comm_vocab` or `view_radius` values just work; `comm_vocab=0` removes the message encoder and the
SAY head.

Parameter sharing: one network for all agents (the observation carries class, faction and agent index; the critic
and the trunk also get an agent-id embedding), or one network per faction with `--per-faction`.

## Turn-based credit assignment

Emberdeep is turn based: in the Parallel / vector API only the decider (`infos[i]["acting"][j]`) has a non-NOOP
action, but *every* agent receives the step's reward. `RolloutCollector` (`examples/rollout.py`) therefore builds, for
every (environment, agent), a trajectory of **its own decisions only**:

```
env step:        1     2     3     4     5     6     7
decider:        A0    A1    A1    B0    A0    A1    A0           (B0 = a foe: scripted, resolved inside step())
A0 transitions: [ d1 ............. ][ d5 ....... ][ d7 .. ]     reward of d1 = r1 + r2 + r3 + r4
                  ^ own decision      (everything that happens until A0 decides again - hits by teammates, the
                                       monsters' turns, healing from others - is summed into the previous decision)
```

* A transition is closed when the agent decides again (bootstrap value = V of the new decision state), when the
  agent dies or the episode terminates (no bootstrap), or when the episode is truncated.
* `gamma` and the GAE `lambda` act per **decision step**, not per env step. The advantage chain stops at episode ends.
* Open transitions at the end of a rollout are carried into the next one, so partially accumulated rewards are never
  trained on. The first decision after a reset starts a new episode's chain.
* Truncation (time limit) is treated as a **timeout = defeat** by default (`--timeout-penalty 1`, no bootstrap); with
  `--timeout-penalty 0` the last decision bootstraps from the value of `final_observation` instead.
* `waiting_obs="stale"` is used for training: only the acting agent's observation is recomputed (about 2x faster
  environments) and waiting agents' observations are never read.
* The MAPPO critic reads `obs["state"]` (`include_state_in_obs=True`) of the acting agent's observation, which is the
  same global state for everybody at that step.

Unit tests (`tests/test_train_example.py`): exact reward bookkeeping on a scripted decider sequence, conservation of the
total credited reward on a real vector-env rollout, GAE values, truncation bootstrap / timeout penalty.

## Free actions and the turn limit

Several abilities (Offer Truce, Parley, ...) are *free*: they never end the turn, the game clock stands still and a
policy can repeat them forever. Without a counter-measure PPO finds this "stall forever" optimum within ten updates on
`goblin_ambush_small`: reward 0 beats the expected return of a randomly fighting party and episodes only end at
`max_steps`. The trainer therefore has two safeguards (both are flags; a library-side fix is requested in
`docs/dev/contract_requests/train.md`):

* `--turn-limit 12` (default): after 12 regular decisions without END_TURN the mask of that agent is reduced to
  END_TURN (reaction `DECIDE` choices are never restricted). The restricted mask is what the policy sees and what is
  stored for the PPO ratio, so training stays on-policy. `evaluate.py` applies the same limit.
* `--max-steps 600` and `--timeout-penalty 1`: truncation after 600 env steps and truncated episodes count as defeats.

Measured on `goblin_ambush_small`: without the turn limit the policy collapsed to stalling (entropy below 1 nat, win
rate 0%, ~zero finished episodes per update) within 10-25 updates in every variant we tried (default payoffs, step
penalty, stronger damage shaping, different learning rates and epochs); with the turn limit the same code reaches
70-90% wins within 50k decisions (default payoffs, no shaping).

## Hyper-parameters

| flag | default | notes |
|---|---|---|
| `--algo` | `mappo` | `ippo` = critic on the agent's own observation |
| `--envs` / `--rollout-steps` | 8 / 64 | batch = 512 decisions per update; async worker processes (`--sync` for one in-process env) |
| `--epochs` / `--minibatch` | 4 / 128 | 16 gradient steps per update |
| `--lr` | 3e-4 | Adam (eps 1e-5), linear decay to 2% over the run (`--no-lr-decay`) |
| `--gamma` / `--gae-lambda` | 0.99 / 0.95 | per decision |
| `--clip` / `--vf-clip` | 0.2 / 0.2 | value clipping in normalised units; running value normalisation (`--no-value-norm`) |
| `--ent-coef` | 0.01 | `--ent-coef-final` anneals linearly |
| `--vf-coef`, `--max-grad-norm` | 0.5, 0.5 | |
| `--target-kl` | off | early stop of the epochs |
| `--turn-limit`, `--max-steps`, `--timeout-penalty`, `--step-penalty` | 12, 600, 1.0, 0 | see above |
| `--d-model`, `--hidden`, `--layers`, `--heads` | 64, 256, 2, 4 | network size |
| `--threads` | 2 | torch threads (environment workers use one thread each) |
| `--time-limit MIN`, `--total-steps`, `--max-updates` | -, 2M, - | stopping rules; LR decay follows the larger of step / time progress |

Training environments are created with `waiting_obs="stale"`, `obs_dtype="float16"`, `map_dtype="uint8"` (about 4x
less memory) and `include_state_in_obs` for MAPPO; `--env-kw` overrides anything. `--reward-preset dense` (damage 0.05,
kills 0.2, ...) speeds up learning of combat but changes the payoffs; it is not needed for `goblin_ambush_small`.

## Social-dilemma experiments

All reward knobs of the environment are command-line flags, so sweeps are shell loops:

```bash
# prosocial sweep (r' = (1-a) r_i + a mean(r_teammates)) on the stag hunt
for a in 0.0 0.25 0.5 0.75 1.0; do
  python examples/train_mappo.py --scenario dragon_hoard --prosocial $a --envs 4 --time-limit 30 --name hoard_pro$a
done
# SVO sweep (degrees) with the mean over *all* agents on a competitive map
for th in 0 30 45 60 90; do
  python examples/train_mappo.py --scenario cursed_coffer --svo $th --env-kw svo_group="'all'" --per-faction --name coffer_svo$th
done
# cheap talk and reactions as decisions
python examples/train_mappo.py --scenario cursed_coffer --comm 8 --reaction-mode agent --per-faction
```

`--team-reward` gives every agent its team's mean reward. Compare runs through the social metrics the environment
reports at the end of each episode (all logged as `ep/<name>`, averaged per update): `equality`, `wealth_gini`,
`betrayals_total`, `truces_formed`, `truces_broken`, `friendly_fire`, `heals_given`, `revives`, `gold_given_total`,
`free_riding_index`, `collective_return`, `survival_rate`, plus `scenario/...` metrics. For a clean comparison
evaluate every final checkpoint with the *same* seeds (`evaluate.py ... --episodes 100 --json out.json`) rather than
the training-time averages (training samples actions and the policy changes while the episodes run).

Advice: use the default payoffs when studying dilemmas (do not use `--reward-preset dense`), train with
`--per-faction` for multi-team scenarios, and give the stag-hunt / common-pool scenarios far more than 10 minutes
(`dragon_hoard` and `fountain_of_life` have longer episodes and sparser payoffs than the debugging scenario).

## Logged metrics

`metrics.csv` has one row per update (columns appear as soon as they are first logged):

| column | meaning |
|---|---|
| `env_steps`, `decisions`, `sps`, `time`, `t_rollout`, `t_optim` | progress and speed (env steps / second) |
| `win_rate`, `episodes`, `return/<agent>`, `return_raw/<agent>`, `win/<agent>`, `survived/<agent>` | episodes finished in this update (sampled policy) |
| `ep/steps`, `ep/rounds`, `ep/team_return/<faction>`, `ep/...` | episode length and every global / scenario metric of `docs/ENV_API.md` |
| `entropy`, `kl`, `clipfrac`, `pg_loss`, `v_loss`, `explained_var`, `grad_norm` | PPO diagnostics (suffix `/<policy>` with `--per-faction`) |
| `timeouts`, `lr`, `ent_coef`, `reward_per_decision` | schedule and reward scale |

TensorBoard event files are written next to the CSV when `tensorboard` is importable.

## Expected results and runtimes

Measured on this repository's CI box (4 vCPUs shared with other jobs, `--threads 2`):

**`goblin_ambush_small`** (2 heroes vs 3 goblins; MAPPO, 8 envs, `--env-kw view_radius=6 --time-limit 8`, default payoffs):
173 updates, 89k env steps (= decisions) in 483 s, about 185 env steps/s (a single environment alone does ~450; per
512-step update: 1.5 s rollout + 1.3 s optimisation).

| | win rate | return / agent | episode length (decisions) |
|---|---|---|---|
| random legal actions (100 episodes) | 6% (s.e. 2.4) | -0.74 | 120 |
| trained, sampled actions (100 episodes) | 73% (s.e. 4.4) | +1.27 | 41 |
| trained, deterministic (100 episodes) | 83% (s.e. 3.8) | +1.53 | 32 |
| hand-written NumPy baseline (`pettingzoo_quickstart.py`, 40 episodes) | 75% | | |

Training-time win rate (sampled policy, all episodes finished in the window; the curve is noisy):
39% (first 10k steps), 59% (20k), 71% (30k), 76% (40k), 79% (50k), 80% (70k), 85-88% (80-90k). More time keeps
helping slowly; the deterministic evaluation policy is better than the sampled one.

**`duel`** (two learning factions, `--per-faction`, 4 min, 43k steps, 84 updates): the red policy beats a random blue
opponent in 78% of the episodes (blue 13%), the blue policy beats a random red in 82% (red 0%); random vs random is
57% / 42%. Between themselves the two policies mostly stall (best response is to defend: only 10% / 30% decisive
games and ~470 decisions per episode), which is the expected behaviour of a few minutes of simultaneous self-play.

Only MAPPO was run for the full 8 minutes; IPPO is covered by the smoke test and a short run, but no learning curve
is reported here. Runtime scales with `--envs` up to the number of free cores (the numbers above were taken while other
jobs were using the same 4 vCPUs).

## Watching a trained agent

```bash
python examples/evaluate.py runs/<run>/final.pt --episodes 3 --record out.edr --annotate
python -m emberdeep.viewer out.edr
```

`--annotate` stores, for every decision, `{"value": V(s), "top_actions": [(label, p) x 5], "note": "chose: ..."}`
per agent in the replay frames; the viewer's *Policy insight* panel shows the value gauge and the top-5 action
probabilities with human-readable labels from `env.action_meanings`. (The annotation of a frame describes the
decision that produced it.) `--deterministic` plays argmax actions, `--opponent random` makes learning agents outside
`--faction` play randomly, `--policy random` gives the random baseline.

## Limitations

* Sample efficiency: with ~1.4 ms per env step on CPU, a 10-minute run is ~100-150k decisions. Harder scenarios
  (`goblin_ambush`, `dragon_hoard`, ...) need hours, not minutes; the code scales with `--envs` and `--time-limit`.
* Networks and the PPO loop are deliberately plain (no recurrence, no population training, no opponent pool). Agents
  see their `messages` / social-history features but have no memory beyond them.
* Self-play with `--per-faction` trains all factions simultaneously (non-stationary); there is no league / best-response
  evaluation.
* The turn limit is a trainer-side wrapper around a missing environment feature (see `docs/dev/contract_requests/train.md`);
  results trained with different limits are not directly comparable.
* An episode that is already over after `reset()` forces a restart of all environments (rare, see contract requests).
