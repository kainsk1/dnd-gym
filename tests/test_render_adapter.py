"""World -> render contract: valid StaticScene / FrameState for every scenario, action descriptors, replays,
rgb_array rendering (skipped without a working GL context)."""
import json
import os
import pickle

import numpy as np
import pytest

import emberdeep
from emberdeep.render.api import ActionDescriptor, FrameState, StaticScene
from helpers_env import ALL_SCENARIOS, SMALL, clean_defs  # noqa: F401 (autouse fixture)


def make_frame(env):
    from emberdeep.env.render_adapter import FrameBuilder
    fb = FrameBuilder(env.core)
    return fb, fb.scene(), fb.frame()


@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_adapter_builds_valid_scene_and_frames(scenario):
    env = emberdeep.make_aec(scenario)
    env.reset(seed=1)
    fb, scene, frame = make_frame(env)
    core = env.core
    w = core.world
    assert isinstance(scene, StaticScene) and isinstance(frame, FrameState)
    H, W = w.map.H, w.map.W
    assert (scene.height, scene.width) == (H, W) and scene.terrain.shape == (H, W) == scene.material.shape
    assert scene.terrain.dtype == np.uint8 and scene.agent_ids == list(core.agents)
    assert len(scene.factions) == len(w.factions) and all(len(f.color) == 3 for f in scene.factions)
    assert scene.comm_vocab == core.info.comm_vocab and len(scene.comm_vocab) == 8
    assert scene.theme and len(scene.ambient) == 3 and all(0 <= c <= 1.5 for c in scene.ambient)
    assert scene.regions == {k: [tuple(c) for c in v] for k, v in w.regions.items()}
    for l in scene.lights:
        assert l.bright >= 0 and l.dim >= 0 and 0 <= l.x <= W and 0 <= l.y <= H
    # frame
    ids = [e.id for e in frame.entities]
    assert len(ids) == len(set(ids))
    cre = [e for e in frame.entities if e.kind == "creature"]
    assert len(cre) == len(w.creatures)
    for c in cre:
        assert 0 <= c.hp <= c.max_hp or c.dead
        assert set(("str", "dex", "con", "int", "wis", "cha")) <= set(c.stats)
        assert c.faction == w.entities[c.id].faction
        if c.agent_id:
            assert c.controller == "agent" and any(k.startswith("r_") or k == "return" for k in c.stats)
    assert "omniscient" in frame.visible and all(a in frame.visible for a in core.agents)
    assert all(f"faction:{i}" in frame.visible for i in range(len(w.factions)))
    for k, v in {**frame.visible, **frame.explored}.items():
        assert v.shape == (H, W) and v.dtype == bool
    assert frame.light_level.shape == (H, W)
    assert len(frame.relations) == len(w.factions)
    assert frame.mode in ("exploration", "combat") and not frame.done
    assert set(frame.returns) == set(core.agents) == set(frame.rewards)
    for e in frame.entities:
        if e.kind == "door":
            assert (e.x, e.y) in frame.doors
    pickle.loads(pickle.dumps((scene, frame)))                         # replays need picklable objects


def test_frames_over_an_episode_are_consistent_and_events_serialisable():
    env = emberdeep.make_aec("fountain_of_life")
    env.reset(seed=2)
    fb = __import__("emberdeep.env.render_adapter", fromlist=["FrameBuilder"]).FrameBuilder(env.core)
    rng = np.random.default_rng(0)
    n_events = 0
    for _ in range(80):
        if not env.agents:
            break
        agent = env.agent_selection
        obs, _r, te, tr, _i = env.last()
        if te or tr:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        out = env.core.last_outcome
        fr = fb.frame(out.events, out)
        evs = fb.event_dicts(out.events)
        n_events += len(evs)
        json.dumps(evs)                                                # JSON-serialisable
        assert fr.step == env.core.env_steps and fr.round == env.core.world.round
        for z in fr.zones:
            assert z.cells and isinstance(z.kind, str)
    assert n_events > 20
    assert fr.metrics                                                  # metrics are provided while stepping


def test_dead_agents_stay_in_frames_flagged_dead():
    env = emberdeep.make_aec(SMALL)
    env.reset(seed=1)
    core = env.core
    from emberdeep.core import resolution as R
    c = core.world.entities[core.agent_eid[core.agents[0]]]
    R.die(core.world, c, -1, "test")
    core._invalidate()
    from emberdeep.env.render_adapter import FrameBuilder
    fr = FrameBuilder(core).frame()
    v = [e for e in fr.entities if e.id == c.eid][0]
    assert v.dead and not v.alive and "dead" in v.statuses
    assert not fr.visible[core.agents[0]].any()


def test_say_events_become_frame_messages():
    env = emberdeep.make_aec(SMALL)
    env.reset(seed=0)
    from emberdeep.env.render_adapter import FrameBuilder
    fb = FrameBuilder(env.core)
    env.step(env.core.layout.base_say + 2)
    out = env.core.last_outcome
    fr = fb.frame(out.events, out)
    assert fr.messages and fr.messages[0][1] == 2 and fr.messages[0][2] == env.core.info.comm_vocab[2]


def test_action_meanings_describe_every_legal_action():
    env = emberdeep.make_aec("goblin_ambush")
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(40):
        if not env.agents:
            break
        agent = env.agent_selection
        obs, _r, te, tr, _i = env.last()
        if te or tr:
            env.step(None)
            continue
        descs = env.action_meanings(agent)
        legal = set(np.flatnonzero(obs["action_mask"]).tolist())
        assert {d.index for d in descs} == legal
        for d in descs:
            assert isinstance(d, ActionDescriptor) and d.label
            kinds.add(d.kind)
            if d.kind == "ability":
                assert d.ability_id and d.cost
                if d.hit_chance is not None:
                    assert 0.0 <= d.hit_chance <= 1.0 and d.expected_damage >= 0
            if d.kind == "move":
                assert d.target_cell is not None
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    assert {"end_turn", "move", "ability"} <= kinds


def test_area_previews_for_area_abilities():
    env = emberdeep.make_aec("goblin_ambush")
    env.reset(seed=1)
    core = env.core
    w = core.world
    wiz = [a for a in core.agents if w.entities[core.agent_eid[a]].template_id == "wizard"][0]
    # skip ahead until the wizard is the decider
    for _ in range(40):
        if env.agent_selection == wiz:
            break
        env.step(0)
    assert env.agent_selection == wiz
    descs = env.action_meanings(wiz)
    areas = [d for d in descs if d.area_cells]
    # at least one area ability (burning hands / sleep / fireball...) has a preview inside the map
    for d in areas[:5]:
        assert all(0 <= x < w.map.W and 0 <= y < w.map.H for x, y in d.area_cells)


# ------------------------------------------------------------------------------------------ replays
def test_recorder_roundtrip(tmp_path):
    from emberdeep.replay import Recorder, load_replay
    path = str(tmp_path / "ep.edr")
    env = Recorder(emberdeep.make_aec(SMALL), path)
    env.reset(seed=2)
    rng = np.random.default_rng(0)
    while env.agents:
        agent = env.agent_selection
        obs, _r, te, tr, _i = env.last()
        env.step(None if (te or tr) else int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    env.close()
    assert os.path.exists(path)
    rp = load_replay(path)
    assert isinstance(rp.scene, StaticScene) and len(rp) > 3 and rp.meta["scenario"] == SMALL
    assert rp.meta["seed"] == 2 and rp.frames[-1][0].done
    assert all(isinstance(f, FrameState) and isinstance(e, list) for f, e in rp.frames)
    assert rp.frames[0][1], "initial frame carries the reset events"
    # parallel recording
    path2 = str(tmp_path / "par.edr")
    penv = Recorder(emberdeep.make_parallel(SMALL), path2)
    obs, _ = penv.reset(seed=3)
    while penv.agents:
        obs, *_ = penv.step({a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in penv.agents})
    penv.close()
    assert len(load_replay(path2)) > 2
    with pytest.raises(ValueError):
        bad = tmp_path / "x.edr"
        import gzip
        with gzip.open(bad, "wb") as f:
            pickle.dump({"nope": 1}, f)
        load_replay(str(bad))


# ------------------------------------------------------------------------------------------ rendering
def _gl_or_skip(quality="low", size=(480, 270)):
    os.environ.setdefault("DISPLAY", ":99")
    try:
        from emberdeep.render import Renderer
        r = Renderer("rgb_array", size[0], size[1], quality, hud=True, audio=False)
        r.close()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no usable OpenGL context: {e}")


def test_rgb_array_render_is_nontrivial():
    _gl_or_skip()
    env = emberdeep.make_aec("goblin_ambush", render_mode="rgb_array", render_width=480, render_height=270,
                             render_quality="low", render_sound=False)
    env.reset(seed=1)
    img0 = env.render()
    assert img0.shape == (270, 480, 3) and img0.dtype == np.uint8
    assert img0.std() > 10 and len(np.unique(img0.reshape(-1, 3), axis=0)) > 200
    rng = np.random.default_rng(0)
    for _ in range(6):
        obs, *_ = env.last()
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
    img1 = env.render()
    assert img1.shape == img0.shape and img1.std() > 10
    assert not np.array_equal(img0, img1)                                  # the game moved on
    env.set_annotations({env.possible_agents[0]: {"value": 0.5, "top_actions": [("End turn", 0.9)], "note": "x"}})
    assert env.render().shape == img0.shape
    env.close()


def test_render_modes_none_has_no_renderer_and_parallel_gym_render():
    env = emberdeep.make_aec(SMALL)
    env.reset(seed=0)
    assert env.render() is None
    assert env._renderer is None
    _gl_or_skip()
    g = emberdeep.make_gym(SMALL, render_mode="rgb_array", render_width=320, render_height=180, render_quality="low",
                           render_sound=False)
    g.reset(seed=0)
    assert g.render().shape == (180, 320, 3)
    g.close()
    p = emberdeep.make_parallel(SMALL, render_mode="rgb_array", render_width=320, render_height=180,
                                render_quality="low", render_sound=False)
    p.reset(seed=0)
    assert p.render().shape == (180, 320, 3)
    p.close()


def test_viewer_exports_frames(tmp_path):
    _gl_or_skip()
    from emberdeep.replay import Recorder, load_replay
    from emberdeep.viewer import export_frames
    path = str(tmp_path / "v.edr")
    env = Recorder(emberdeep.make_aec(SMALL), path)
    env.reset(seed=2)
    n = 0
    while env.agents and n < 6:
        obs, _r, te, tr, _i = env.last()
        if te or tr:
            env.step(None)
            continue
        env.step(int(np.flatnonzero(obs["action_mask"])[0]))
        n += 1
    env.save()
    env.close()
    rp = load_replay(path)
    paths = export_frames(rp, str(tmp_path / "frames"), (320, 180), "low", stop=3)
    assert len(paths) == 3 and all(os.path.getsize(p) > 1000 for p in paths)
    from emberdeep.viewer import main
    assert main([path, "--info"]) == 0
