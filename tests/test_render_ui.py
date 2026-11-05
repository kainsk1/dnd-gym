"""Tests for the HUD (render/ui) and the procedural audio engine. Headless (SDL dummy drivers)."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from emberdeep.core.events import EventType as ET  # noqa: E402
from emberdeep.render import assets  # noqa: E402
from emberdeep.render.api import ActionDescriptor  # noqa: E402
from emberdeep.render.hud_api import UICommand  # noqa: E402
from emberdeep.render.ui import _mock_frames as M  # noqa: E402
from emberdeep.render.ui.hud import HUD  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _pg():
    pygame.init()
    pygame.display.set_mode((64, 64))
    yield
    pygame.quit()


def make_hud(size=(1600, 900), stage="combat", **view_kw):
    scene = M.make_scene()
    frame = M.make_frame(stage)
    hud = HUD()
    hud.set_scene(scene)
    hud.on_frame(frame)
    view = M.make_view(size, frame, scene, **view_kw)
    surf = pygame.Surface(size, pygame.SRCALPHA)
    return hud, scene, frame, view, surf


def run_frames(hud, surf, view, n=3, dt=1 / 30):
    for _ in range(n):
        hud.update(dt, view)
        surf.fill((0, 0, 0, 0))
        hud.draw(surf, view)


def kd(key, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod, unicode="")


def click(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


# ------------------------------------------------------------------------------ events
def test_every_event_type_has_a_sample():
    kinds = {d["t"] for d in M.all_event_dicts()}
    assert kinds == {int(t) for t in ET}


def test_hud_handles_every_event_type():
    hud, scene, frame, view, surf = make_hud()
    for d in M.all_event_dicts():
        hud.on_event(d, frame, view)
        run_frames(hud, surf, view, 1)
    run_frames(hud, surf, view, 30, dt=0.1)
    assert len(hud.log.entries) > 30


def test_hud_tolerates_sparse_events():
    hud, scene, frame, view, surf = make_hud()
    for t in ET:
        hud.on_event({"t": int(t), "type": t.name}, frame, view)           # no src/dst/data at all
        hud.on_event({"t": int(t), "src": 999, "dst": 998, "amount": 0, "kind": 99, "data": {}}, frame, view)
    hud.on_event({"t": 9999}, frame, view)
    run_frames(hud, surf, view, 3)


def test_full_mock_fight_replays():
    fight = M.MockFight()
    hud = HUD()
    hud.set_scene(fight.scene)
    surf = pygame.Surface((1600, 900), pygame.SRCALPHA)
    for frame, events in fight.steps:
        view = M.make_view((1600, 900), frame, fight.scene)
        hud.on_frame(frame)
        for e in events:
            hud.on_event(e, frame, view)
        run_frames(hud, surf, view, 5, dt=0.2)


# ------------------------------------------------------------------------------ drawing
@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080), (2560, 1440), (800, 600)])
def test_draws_at_resolutions(size):
    hud, scene, frame, view, surf = make_hud(size, selected=4, hovered=8,
                                             options={"research": True, "labels": True})
    for d in M.all_event_dicts()[:20]:
        hud.on_event(d, frame, view)
    run_frames(hud, surf, view, 4)
    assert surf.get_bounding_rect().w > 100               # something was drawn
    # centre of the screen is mostly empty: the HUD must not cover the battlefield
    if size[0] >= 1280:
        cx, cy = size[0] // 2, size[1] // 2
        assert surf.get_at((cx, cy + size[1] // 5))[3] < 200


def test_draw_without_scene_or_frame_is_safe():
    hud = HUD()
    surf = pygame.Surface((800, 600), pygame.SRCALPHA)
    scene = M.make_scene()
    view = M.make_view((800, 600), M.make_frame(), scene)
    hud.update(0.016, view)
    hud.draw(surf, view)
    hud.handle_event(kd(pygame.K_SPACE), view)
    hud.world_click((1, 1), -1, 1, view)
    hud.set_scene(scene)
    hud.draw(surf, view)


def test_exploration_and_pov_modes():
    for stage in ("exploration", "combat"):
        for pov in ("omniscient", "fighter", "faction:1"):
            hud, scene, frame, view, surf = make_hud(stage=stage, pov=pov)
            run_frames(hud, surf, view, 3)


def test_draw_time_budget():
    hud, scene, frame, view, surf = make_hud()
    for d in M.all_event_dicts()[:25]:
        hud.on_event(d, frame, view)
    run_frames(hud, surf, view, 20)
    ts = []
    for _ in range(30):
        hud.update(1 / 60, view)
        surf.fill((0, 0, 0, 0))
        t0 = time.perf_counter()
        hud.draw(surf, view)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    # target is 3 ms on a laptop; CI machines are slow, so only guard against gross regressions
    assert ts[len(ts) // 2] < 0.05


def test_banners_scoreboard_and_help():
    hud, scene, frame, view, surf = make_hud()
    frame.done = True
    frame.outcome = "All goblins defeated"
    hud.on_frame(frame)
    hud.on_event(M.ev(ET.EPISODE_END, reason="done", winners=[0]), frame, view)
    assert hud.final_kind == "victory"
    run_frames(hud, surf, view, 90, dt=0.05)
    assert hud.scoreboard.shown
    assert hud.handle_event(kd(pygame.K_ESCAPE), view) is not None      # dismisses
    assert not hud.scoreboard.shown
    hud.handle_event(kd(pygame.K_F1), view)
    assert hud.help_open
    run_frames(hud, surf, view, 2)
    hud.handle_event(kd(pygame.K_F1), view)
    assert not hud.help_open
    hud.on_event(M.ev(ET.EPISODE_END, reason="wiped", winners=[1]), frame, view)
    assert hud.final_kind == "defeat"


# ------------------------------------------------------------------------------ input
def test_hotkeys_emit_commands():
    hud, scene, frame, view, surf = make_hud()
    run_frames(hud, surf, view)
    assert hud.handle_event(kd(pygame.K_SPACE), view).kind == "toggle_pause"
    assert hud.handle_event(kd(pygame.K_n), view).kind == "step"
    c = hud.handle_event(kd(pygame.K_RIGHTBRACKET), view)
    assert (c.kind, c.value) == ("speed", 2.0)
    c = hud.handle_event(kd(pygame.K_v), view)
    assert c.kind == "pov" and c.value == "fighter"
    c = hud.handle_event(kd(pygame.K_g), view)
    assert (c.kind, c.value) == ("toggle", "grid")
    assert hud.handle_event(kd(pygame.K_TAB), view).kind == "skip_anim"
    assert hud.handle_event(kd(pygame.K_z), view) is None              # unknown key is not consumed


def test_mouse_over_panels_is_consumed_and_world_is_not():
    hud, scene, frame, view, surf = make_hud()
    run_frames(hud, surf, view)
    assert hud.handle_event(click((20, 100)), view) is not None         # on a party card
    assert hud.handle_event(click((800, 650)), view) is None            # open battlefield
    # wheel over the log scrolls it and is consumed; over the world it is not
    for i in range(60):
        hud.on_event(M.ev(ET.LOG, text=f"line {i}"), frame, view)
    run_frames(hud, surf, view)
    r = hud.log.rect
    hud.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=r.center, rel=(0, 0), buttons=(0, 0, 0)), view)
    ev = pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=3)
    assert hud.handle_event(ev, view) is not None
    assert hud.log.scroll > 0
    hud.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(800, 650), rel=(0, 0), buttons=(0, 0, 0)), view)
    assert hud.handle_event(ev, view) is None


def test_controls_buttons_and_pov_popup():
    hud, scene, frame, view, surf = make_hud()
    run_frames(hud, surf, view)
    r = hud.controls.rect
    # speed chip row is the second row; find any hotspot whose click returns a speed command
    cmds = []
    for h in list(hud.hots):
        if h.click and r.contains(h.rect):
            res = None
            try:
                res = h.click()
            except Exception:  # noqa: BLE001
                pass
            if isinstance(res, UICommand):
                cmds.append(res.kind)
    assert {"toggle_pause", "step", "skip_anim", "speed", "toggle"} <= set(cmds)
    hud.controls.pov_open = True
    run_frames(hud, surf, view)
    ir = hud.controls.rects["pov:1"]
    cmd = hud.handle_event(click(ir.center), view)
    assert cmd.kind == "pov" and cmd.value == view.pov_options[1]
    assert not hud.controls.pov_open


def test_minimap_click_focuses():
    hud, scene, frame, view, surf = make_hud()
    run_frames(hud, surf, view)
    mr = hud.minimap.map_rect
    cmd = hud.handle_event(click(mr.center), view)
    assert cmd.kind == "focus" and isinstance(cmd.value, tuple)
    assert abs(cmd.value[0] - scene.width / 2) < 2


def test_select_and_focus_via_world_click():
    hud, scene, frame, view, surf = make_hud()
    c = hud.world_click((14, 8), 5, 1, view)
    assert (c.kind, c.value) == ("select", 5)
    c2 = hud.world_click((14, 8), 5, 1, view)
    assert c2.kind == "focus"
    c3 = hud.world_click((3, 3), -1, 1, M.make_view((1600, 900), frame, scene, selected=5))
    assert (c3.kind, c3.value) == ("select", -1)


def test_inspector_opens_for_selection():
    hud, scene, frame, view, surf = make_hud(selected=4)
    run_frames(hud, surf, view)
    assert hud.inspector.rect.h > 100
    # close button / Esc
    cmd = hud.handle_event(kd(pygame.K_ESCAPE), view)
    assert cmd.kind == "select" and cmd.value == -1


# ------------------------------------------------------------------------------ human play
def human_hud(**kw):
    hud, scene, frame, view, surf = make_hud(human=True, **kw)
    acts = M.make_actions()
    hud.begin_human_choice("rogue", acts)
    run_frames(hud, surf, view)
    return hud, scene, frame, view, surf, acts


def test_human_end_turn_by_key_and_click():
    hud, scene, frame, view, surf, acts = human_hud()
    c = hud.handle_event(kd(pygame.K_RETURN), view)
    assert (c.kind, c.value) == ("choose_action", 0)
    r = hud.actionbar.rects["end"]
    c = hud.handle_event(click(r.center), view)
    assert (c.kind, c.value) == ("choose_action", 0)


def test_human_self_ability_chooses_immediately_and_targeted_enters_target_mode():
    hud, scene, frame, view, surf, acts = human_hud()
    groups = {g.label: g for g in hud.actionbar.groups}
    # Dash has no target -> immediate
    c = hud.handle_event(kd(groups["Dash"].code), view)
    assert (c.kind, c.value) == ("choose_action", 5)
    # Shortsword has two targets -> target mode, nothing chosen yet
    c = hud.handle_event(kd(groups["Shortsword"].code), view)
    assert c is not None and c.kind != "choose_action"
    assert hud.actionbar.sel is groups["Shortsword"]
    run_frames(hud, surf, view)
    # clicking the second goblin picks the descriptor for that target
    c = hud.world_click((14, 11), 8, 1, view)
    assert (c.kind, c.value) == ("choose_action", 2)
    assert hud.actionbar.sel is None


def test_human_target_mode_cancel_with_esc_and_right_click():
    hud, scene, frame, view, surf, acts = human_hud()
    g = next(g for g in hud.actionbar.groups if g.label == "Shortbow")
    hud.handle_event(kd(g.code), view)
    assert hud.actionbar.sel is g
    assert hud.handle_event(kd(pygame.K_ESCAPE), view) is not None
    assert hud.actionbar.sel is None
    hud.handle_event(kd(g.code), view)
    assert hud.world_click((1, 1), -1, 3, view) is not None
    assert hud.actionbar.sel is None


def test_human_button_click_and_aoe_cell_target():
    hud, scene, frame, view, surf, acts = human_hud()
    g = next(g for g in hud.actionbar.groups if g.label == "Fireball")
    r = hud.actionbar.rects["g:" + g.key]
    c = hud.handle_event(click(r.center), view)
    assert c is not None and c.kind == "noop"
    assert hud.actionbar.sel is g
    c = hud.world_click((15, 10), -1, 1, view)
    assert (c.kind, c.value) == ("choose_action", 9)


def test_human_floor_click_picks_nearest_move():
    hud, scene, frame, view, surf, acts = human_hud()
    c = hud.world_click((12, 7), -1, 1, view)
    assert (c.kind, c.value) == ("choose_action", 12)
    c = hud.world_click((10, 7), -1, 1, view)
    assert (c.kind, c.value) == ("choose_action", 14)
    c = hud.handle_event(kd(pygame.K_KP8), view)
    assert (c.kind, c.value) == ("choose_action", 11)


def test_human_token_click_without_selection_attacks_or_approaches():
    hud, scene, frame, view, surf, acts = human_hud()
    c = hud.world_click((14, 8), 5, 1, view)            # first ability descriptor targeting eid 5
    assert (c.kind, c.value) == ("choose_action", 1)


def test_disabled_ability_does_nothing():
    hud, scene, frame, view, surf, acts = human_hud()
    g = next(g for g in hud.actionbar.groups if g.label == "Second Wind")
    assert not g.ok
    assert hud.handle_event(kd(g.code), view) is not None  # consumed ...
    run_frames(hud, surf, view)
    r = hud.actionbar.rects["g:" + g.key]
    c = hud.handle_event(click(r.center), view)
    assert c is None or c.kind != "choose_action"


def test_say_menu():
    hud, scene, frame, view, surf, acts = human_hud()
    hud.actionbar.say_open = True
    run_frames(hud, surf, view)
    ir = hud.actionbar.rects["s:1"]
    res = hud.handle_event(click(ir.center), view)
    assert (res.kind, res.value) == ("choose_action", 17)


def test_reaction_modal_choices():
    hud, scene, frame, view, surf = make_hud(human=True)
    frame.decision = {"eid": 3, "kind": 3, "options": ["Uncanny Dodge", "Take the hit"]}
    hud.on_frame(frame)
    A = ActionDescriptor
    hud.begin_human_choice("rogue", [A(40, "decision", "Uncanny Dodge", icon="shield-bounces", cost="reaction"),
                                     A(41, "decision", "Take the hit")])
    run_frames(hud, surf, view)
    c = hud.handle_event(kd(pygame.K_2), view)
    assert (c.kind, c.value) == ("choose_action", 41)
    r = hud.actionbar.rects["d:0"]
    c = hud.handle_event(click(r.center), view)
    assert (c.kind, c.value) == ("choose_action", 40)
    hud.end_human_choice()
    run_frames(hud, surf, view)
    assert not hud.actionbar.active


# ------------------------------------------------------------------------------ rules of the HUD
def test_enemy_hp_is_bucketed_in_agent_pov_and_exact_for_allies():
    hud, scene, frame, view, surf = make_hud(pov="fighter")
    bug = hud.ent(8)
    frac, label = hud.hp_display(bug, view)
    assert label in ("Unhurt", "Scratched", "Wounded", "Bloodied", "Near death")
    ally = hud.ent(2)
    assert hud.hp_display(ally, view)[1] is None
    omni = M.make_view((1600, 900), frame, scene, pov="omniscient")
    assert hud.hp_display(bug, omni)[1] is None
    tip = hud.entity_tip(bug, view)
    assert "AC ?" in tip.lines[0][0]


def test_ui_icons_referenced_in_source_exist():
    root = Path(__file__).resolve().parents[1] / "emberdeep" / "render" / "ui"
    pat = re.compile(r'"([a-z0-9]+(?:-[a-z0-9]+)+)"')
    missing = set()
    for p in root.glob("*.py"):
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            key = m.group(1)
            if not assets.icon_exists(key):
                missing.add((p.name, key))
    assert not missing, missing


def test_scale_follows_screen_height():
    hud = HUD()
    hud.ui.set_size(1280, 720)
    s1 = hud.ui.scale
    hud.ui.set_size(2560, 1440)
    assert hud.ui.scale > s1 * 1.9


# ------------------------------------------------------------------------------ audio
def test_audio_disabled_is_noop():
    from emberdeep.render.audio import AudioEngine
    a = AudioEngine(enabled=False)
    assert not a.ok
    for d in M.all_event_dicts():
        a.on_event(d, pan=0.3)
    a.update(0.1)
    a.set_volume(0.2)
    a.toggle_mute()
    a.on_ui("click")
    a.close()


def test_audio_survives_mixer_failure(monkeypatch):
    from emberdeep.render import audio
    if pygame.mixer.get_init():
        pygame.mixer.quit()

    def boom(*a, **k):
        raise pygame.error("no audio device")

    monkeypatch.setattr(pygame.mixer, "init", boom)
    a = audio.AudioEngine(enabled=True)
    assert not a.ok
    for d in M.all_event_dicts():
        a.on_event(d)
    a.update(0.5)
    a.play("clang")
    a.set_volume(1.0)
    assert a.toggle_mute() in (True, False)
    a.close()


def test_audio_plays_with_dummy_device(monkeypatch, tmp_path):
    from emberdeep.render import audio
    monkeypatch.setenv("EMBERDEEP_CACHE", str(tmp_path))
    a = audio.AudioEngine(enabled=True, volume=0.5)
    if not a.ok:                                          # platforms without even a dummy driver
        pytest.skip("no mixer available")
    for d in M.all_event_dicts():
        a.on_event(d, pan=-0.4)
    for _ in range(20):
        a.update(0.05)
    a.on_event(M.ev(ET.COMBAT_START), 0.0)
    for _ in range(60):
        a.update(0.1)
    assert a._music_vol["combat"] > 0.3
    a.on_event(M.ev(ET.COMBAT_END), 0.0)
    for _ in range(80):
        a.update(0.1)
    assert a._music_vol["combat"] < 0.05
    assert (tmp_path / f"sfx_v{audio.CACHE_VERSION}.npz").exists()
    a.close()
    a.on_event(M.ev(ET.ATTACK))                           # after close: still silent no-op


def test_audio_bank_is_clean():
    import numpy as np
    from emberdeep.render import audio
    bank = audio.synth_bank()
    assert set(audio.SFX) <= set(bank)
    for k, v in bank.items():
        assert v.dtype == np.int16 and len(v) > 100 and abs(v).max() > 1000, k
    amb = bank["ambient"].astype(float)
    assert abs(amb[0] - amb[-1]) < 20 * np.abs(np.diff(amb)).mean() + 200       # seamless loop
