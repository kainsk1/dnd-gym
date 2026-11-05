"""Differential tests for the fast observation path (frame table, slot selection, self / entity / ability / map
builders): every optimised function must equal its ``*_ref`` twin bit for bit (values *and* dtype) on states sampled
from random rollouts over all scenarios and several ``EnvConfig`` variants."""
import numpy as np
import pytest

import emberdeep
from emberdeep.core.types import EntityKind
from emberdeep.env import actions as A
from emberdeep.env import kernels as K
from emberdeep.env.frame import FrameTable, FrameTableRef
from helpers_env import clean_defs  # noqa: F401 (autouse fixture)

SCENARIOS = emberdeep.list_scenarios()
FRAME_COLS = ("eid", "kind", "x", "y", "z", "size", "faction", "present", "hp", "max_hp", "temp", "ac", "cond",
              "status", "ctype", "size_enum", "level", "conc", "agent_idx", "initiative", "row_of", "is_agent", "dead",
              "downed", "xp")
CONFIGS = [
    dict(),
    dict(view_radius=5, max_entities=8),
    dict(max_entities=24, enemy_hp="exact", entity_order="fixed"),
    dict(enemy_hp="hidden", social_features=False, allies_always_visible=False),
    dict(vision_sharing="team", comm_vocab=0),
    dict(obs_dtype="float16", map_dtype="uint8", include_state_in_obs=True),
    dict(fog_of_war_map=True, map_dtype="float16"),
]


def _same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), what


def check_state(core, agent, tag=""):
    """Compare fast vs reference for one agent at the current world version; returns 1."""
    fr = core.frame()
    w = core.world
    ob = core.obs_builder
    # --- frame table
    ref = FrameTableRef(w, core.agent_index_of_eid)
    assert fr.n == ref.n and fr.nc == ref.nc and fr.ents == ref.ents
    for name in FRAME_COLS:
        _same(getattr(fr, name), getattr(ref, name), (tag, "frame", name))
    cb, sb = fr.bit_tables()
    rcb, rsb = ref.bit_tables()
    _same(cb, rcb, "cond bits")
    _same(sb, rsb, "status bits")
    if fr.n > fr.nc:
        c_obj = ob._ex["obj_open"]
        for j, e in enumerate(fr.ents[fr.nc:]):
            row = np.zeros(len(ob.fe), np.float32)
            ob._object_row(row, e)
            np.clip(row, -4.0, 4.0, out=row)
            assert row[:c_obj].sum() == 0 and np.array_equal(row[c_obj:], fr.objf[j]), (tag, "objf", j)
    # --- slots
    ctx = core.context(agent)
    se, sr, sv = A.select_slots_ref(core, fr, agent)
    _same(ctx.slot_eids, se, (tag, "slot_eids"))
    _same(ctx.slot_rows, sr, (tag, "slot_rows"))
    _same(ctx.visible_rows, sv, (tag, "visible"))
    # --- builders
    for name in ("_self_vec", "_abilities", "_map"):
        args = (agent, ctx) if name == "_abilities" else (agent, ctx, fr)
        a = getattr(ob, name)(*args)
        b = getattr(ob, name + "_ref")(*args)
        for x, y in zip(a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)):
            _same(x, y, (tag, name))
    for x, y in zip(ob._entities(agent, ctx, fr), ob._entities_ref(agent, ctx, fr)):
        _same(x, y, (tag, "_entities"))
    new = ob.build(agent, ctx, fr)
    old = ob.build_ref(agent, ctx, fr)
    assert new.keys() == old.keys()
    for k in old:
        _same(new[k], old[k], (tag, "obs", k))
    return 1


def _walk(scn, seed, steps, **kw):
    env = emberdeep.make_aec(scn, **kw)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    n = checked = 0
    while env.agents and n < steps:
        obs, _r, te, tr, _ = env.last()
        if te or tr:
            env.step(None)
            continue
        core = env.core
        for ag in env.agents:
            checked += check_state(core, ag, (scn, seed, n, ag))
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        n += 1
    return checked


@pytest.mark.parametrize("scn", SCENARIOS)
def test_fast_path_matches_reference(scn):
    total = sum(_walk(scn, s, 70) for s in range(2))
    assert total > 40


@pytest.mark.parametrize("kw", CONFIGS[1:], ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()))
def test_fast_path_matches_reference_configs(kw):
    total = _walk("goblin_ambush", 1, 60, **kw) + _walk("skirmish_2v2", 2, 40, **kw)
    assert total > 40


@pytest.mark.parametrize("kw", [dict(), dict(flatten_obs=True, obs_dtype="float16"), dict(vision_sharing="team")],
                         ids=["default", "flat", "team"])
def test_reference_mode_rollout_is_identical(kw, monkeypatch):
    """Whole rollouts with ``EMBERDEEP_OBS_REF`` semantics (``kernels.REF``) produce the same observations."""
    def run(ref):
        monkeypatch.setattr(K, "REF", ref)
        env = emberdeep.make_aec("goblin_ambush", **kw)
        rng = np.random.default_rng(3)
        env.reset(seed=3)
        out = []
        for _ in range(60):
            if not env.agents:
                break
            obs, _r, te, tr, _ = env.last()
            if te or tr:
                env.step(None)
                continue
            out.append({k: np.array(v) for k, v in obs.items()})
            env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        return out
    a = run(False)
    b = run(True)
    assert len(a) == len(b) > 20
    for oa, ob_ in zip(a, b):
        assert oa.keys() == ob_.keys()
        for k in oa:
            _same(oa[k], ob_[k], k)


def test_unusual_observers_use_the_reference_selection():
    """Observers outside the fast path (special senses, big size, blinded ...) fall back to ``select_slots_ref``."""
    env = emberdeep.make_aec("goblin_ambush")
    env.reset(seed=0)
    core = env.core
    w = core.world
    for ag in env.agents:
        c = w.entities[core.agent_eid[ag]]
        for patch in ({"darkvision": 60}, {"blindsight": 30}, {"truesight": 60}, {"tremorsense": 30},
                      {"see_invisible": 1}):
            old = dict(c.senses)
            c.senses.update(patch)
            w.vis_dirty()
            core._invalidate()
            fr = core.frame()
            got = A.select_slots(core, fr, ag)
            want = A.select_slots_ref(core, fr, ag)
            for x, y in zip(got, want):
                _same(x, y, (ag, patch))
            c.senses.clear()
            c.senses.update(old)
            w.vis_dirty()
            core._invalidate()


def test_kernels_python_mode_matches_compiled():
    """The kernels stay valid plain Python (``EMBERDEEP_NO_NUMBA=1``): ``.py_func`` equals the compiled result."""
    env = emberdeep.make_aec("goblin_ambush")
    env.reset(seed=0)
    core = env.core
    w = core.world
    for _ in range(6):
        env.step(int(np.flatnonzero(env.last()[0]["action_mask"])[0]))
    ag = env.agents[0]
    fr = core.frame()
    c = w.entities[core.agent_eid[ag]]
    m = w.map
    args = (fr.tab, fr.row(c.eid), w.rules.sight_radius, core.visible_for(ag), m.opaque, w.light_grid(), m.zone_heavy,
            m.zone_dark, 0, False, 8, 6, c.eid << 12, c.faction, True, core.cfg.max_entities, True)
    fast = K.select_all(*args)
    slow = getattr(K.select_all, "py_func", K.select_all)(*args)
    n_ev = int(fast[7])
    assert n_ev == int(slow[7]) and int(fast[8]) == int(slow[8])
    for i, (x, y) in enumerate(zip(fast, slow)):
        x, y = np.asarray(x), np.asarray(y)
        if i in (5, 6):                      # event buffers: only the first n_ev entries are defined
            x, y = x[:n_ev], y[:n_ev]
        assert np.array_equal(x, y), i
