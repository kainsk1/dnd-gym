# Emberdeep v1 — how to test it on your machine

A 20-minute checklist that exercises every part of version 1. Commands assume the repository root.

## 0. Install

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[all]"                # numba, pygame-ce, moderngl, pillow, torch, pytest
```

On Linux, `pip` installs the CUDA build of PyTorch (~2.5 GB). For a CPU-only machine install the small
CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then the line above.
Without the training examples you can skip torch entirely: `pip install -e ".[fast,render,dev]"`.

The first run compiles numba kernels (~10–15 s) and generates procedural textures and sounds
(~5 s); both are cached in `~/.cache/emberdeep` (set `EMBERDEEP_CACHE` to move it).

## 1. Test suite (≈ 15–30 min on a laptop; GL tests need a display)

```bash
pytest -q                               # everything
pytest -q -k "not monsters_play and not spells_all and not archetypes_play"   # fast subset (~5 min)
```

## 2. Watch a game (render mode)

```bash
python -m emberdeep.play --scenario goblin_ambush --watch          # scripted heroes vs goblin ambush
python -m emberdeep.play --scenario dragon_hoard --watch --seed 3  # stag-hunt lair, dark theme + lava
python -m emberdeep.play --scenario fountain_of_life --watch       # monster waves, common-pool fountain
python -m emberdeep.play --scenario deep_delve --watch             # procedural dungeon crawl
```

Controls: mouse wheel zoom, right-drag / WASD pan, `Space` pause, `N` step, `[` `]` speed, `V` cycle
point of view (omniscient / each agent / each faction — fog of war), `F1` help with every hotkey,
`F12` screenshot, `M` minimap, `C` log, `F2` research overlay, `F4` mute. Click a token to inspect it.

## 3. Play it yourself

```bash
python -m emberdeep.play --scenario goblin_ambush --control party_0
```

You control the fighter (others are hero bots). On your turn the action bar shows your abilities
(hotkeys 1–0, Q–P); pick one, then click a highlighted target; click a floor square to move;
`Enter` ends the turn.

## 4. RL API

```bash
python examples/pettingzoo_quickstart.py          # AEC + Parallel + vector env tour, numpy baseline vs random
python examples/random_rollout.py --scenario dragon_hoard --episodes 3 --prosocial 0.3
python -m emberdeep.bench --scenario goblin_ambush --envs 4
```

## 5. Train an agent (≈ 10 min on CPU)

```bash
python examples/train_mappo.py --scenario goblin_ambush_small --algo mappo --envs 8 \
       --env-kw view_radius=6 --time-limit 8 --name goblins
python examples/evaluate.py runs/goblins/final.pt --episodes 100                 # expect ~70–85% win vs ~6% random
python examples/evaluate.py runs/goblins/final.pt --episodes 2 --record g.edr --annotate
python -m emberdeep.viewer g.edr                                                 # F2 shows value + top actions
```

## 6. Replays & video frames

```bash
python -m emberdeep.viewer g.edr --export-frames frames/ --size 1920x1080
```

## What to report back

* Anything that crashes (with the command and traceback).
* Rules you believe are wrong (scenario, round, combat-log lines).
* Visual issues (screenshot via F12) and the GPU / OS you're on.
* Throughput numbers from step 4 on your machine.
