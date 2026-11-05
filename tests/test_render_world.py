"""World renderer smoke tests (skipped when no OpenGL 3.3 context can be created)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygame")
pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def renderer():
    from emberdeep.render.renderer import Renderer
    try:
        r = Renderer("rgb_array", 640, 360, "low", hud=False, audio=False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no GL context: {e}")
    yield r
    r.close()


def test_import_is_lazy():
    import subprocess, sys
    code = "import sys, emberdeep.render; assert 'pygame' not in sys.modules and 'moderngl' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_mock_scene_renders(renderer):
    from emberdeep.render.mock import MockScript
    s = MockScript()
    renderer.set_scene(s.scene)
    steps = s.steps()
    renderer.push(steps[0][1], steps[0][2])
    renderer.play_until_idle(5.0)
    img = renderer.capture()
    assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    assert img.std() > 12
    lum = img.mean(2)
    assert (lum < 25).mean() > 0.02 and (lum > 90).mean() > 0.005   # dark and lit regions


def test_full_script_events(renderer):
    from emberdeep.render.mock import MockScript
    s = MockScript()
    renderer.set_scene(s.scene)
    for label, frame, evs in s.steps():
        renderer.push(frame, evs)
        renderer.play_until_idle(40.0)
    assert renderer.director.idle
    assert renderer.capture().shape == (360, 640, 3)


def test_every_event_type_and_vfx_key(renderer):
    from emberdeep.core.events import EventType
    from emberdeep.render.api import VFX_KEYS
    from emberdeep.render.mock import MockScript
    s = MockScript()
    renderer.set_scene(s.scene)
    f = s.frame()
    renderer.push(f, [])
    evs = []
    for t in EventType:
        evs.append({"t": int(t), "type": t.name, "src": 1, "dst": 11, "amount": 3, "kind": 0, "step": 1, "round": 1, "data": {}})
    for k in VFX_KEYS:
        evs.append({"t": int(EventType.SPELL_CAST), "type": "SPELL_CAST", "src": 2, "dst": 11, "amount": 1, "kind": -1, "step": 1, "round": 1,
                    "data": {"vfx": k, "targets": [11, 12], "point": (16, 8), "spell": k}})
        evs.append({"t": int(EventType.ATTACK), "type": "ATTACK", "src": 1, "dst": 11, "amount": 0, "kind": 0, "step": 1, "round": 1,
                    "data": {"vfx": k, "hit": bool(len(evs) % 2), "crit": len(evs) % 5 == 0}})
    renderer.push(f, evs)
    renderer.play_until_idle(600.0)
    assert renderer.director.idle
    assert renderer.capture().shape == (360, 640, 3)


def test_recipe_for_every_key():
    from emberdeep.render.api import VFX_KEYS
    from emberdeep.render.vfx.recipes import RECIPES
    assert [k for k in VFX_KEYS if k not in RECIPES] == []
