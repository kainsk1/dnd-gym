# Emberdeep examples

| file | what it shows |
|---|---|
| `random_rollout.py` | smallest end-to-end use of the AEC / Parallel API with random legal actions |
| `pettingzoo_quickstart.py` | observation, masks, reward components, a hand-written NumPy baseline, Gymnasium and vector env |
| `train_mappo.py` | **reference MAPPO / IPPO trainer** (PyTorch, CPU), CSV (+ TensorBoard) logging, checkpoints |
| `evaluate.py` | evaluate a checkpoint (or a random policy), record `.edr` replays with value / top-5 action overlay |
| `models.py` | structured network: map CNN + entity transformer + ability encoder + pointer-style factored action head |
| `rollout.py` | turn-based trajectory collector (decision-only transitions, summed waiting rewards), GAE, turn limiter |

The trainer needs `torch` (CPU is enough), `numpy`, `gymnasium`, `pettingzoo`; `tensorboard` is optional. Full
background (architecture, credit assignment, hyper-parameters, experiments): [`docs/TRAINING.md`](../docs/TRAINING.md).

## Quick start

```bash
# 1. train 2 heroes vs 3 goblins (MAPPO, 8 environment processes) for 8 minutes on a laptop CPU
python examples/train_mappo.py --scenario goblin_ambush_small --algo mappo --envs 8 \
       --env-kw view_radius=6 --time-limit 8 --name goblins

# 2. evaluate: 100 episodes against the scripted goblins, and the random-policy baseline
python examples/evaluate.py runs/goblins/final.pt --episodes 100
python examples/evaluate.py --policy random --scenario goblin_ambush_small --episodes 100

# 3. watch it: record a replay with the research overlay (value + top-5 actions) and open the viewer
python examples/evaluate.py runs/goblins/final.pt --episodes 2 --record goblins.edr --annotate
python -m emberdeep.viewer goblins.edr
```

Outputs live in `runs/<name>/`: `metrics.csv` (one row per update), `config.json`, `ckpt_<update>.pt`, `latest.pt`,
`final.pt` (and TensorBoard event files when `tensorboard` is installed; `--no-tensorboard` disables them).

## More commands

```bash
# decentralised critic (IPPO) with 4 envs, 2 torch threads
python examples/train_mappo.py --scenario goblin_ambush --algo ippo --envs 4 --threads 2

# two learning factions, one policy per faction (self-play)
python examples/train_mappo.py --scenario duel --per-faction --envs 8 --time-limit 5
python examples/evaluate.py runs/<run>/final.pt --episodes 50 --opponent random     # policy vs random opponent

# social-dilemma knobs (reward mixing, communication, reactions as decisions)
python examples/train_mappo.py --scenario dragon_hoard --prosocial 0.5 --comm 8 --reaction-mode agent
python examples/train_mappo.py --scenario cursed_coffer --svo 45 --env-kw svo_group="'all'"
python examples/train_mappo.py --scenario goblin_ambush_team --team-reward

# any EnvConfig field: --env-kw key=value (python literal or plain string), repeatable
python examples/train_mappo.py --env-kw "reward_weights={'healing_given':0.02}" --env-kw max_rounds=20

# debugging: one in-process environment
python examples/train_mappo.py --envs 1 --sync --rollout-steps 32 --max-updates 3
```

`python examples/train_mappo.py --help` lists every option. Keep experiments short on shared machines
(`--threads 2`, `--time-limit MINUTES`).

## Tests

```bash
pytest tests/test_train_example.py
```

(2-update smoke runs for MAPPO / IPPO / per-faction, masking, flat-layout assembly vs the env decoder, turn-based
credit assignment, replay with annotations.)
