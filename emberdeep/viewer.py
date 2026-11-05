"""Replay viewer: ``python -m emberdeep.viewer file.edr``.

Plays an ``.edr`` replay (see :mod:`emberdeep.replay`) in the Emberdeep renderer.

Controls (window)
    Space / HUD buttons   pause, resume, speed, step (N), POV (Tab)
    Right / Left          next / previous frame (instant, also while paused)
    Home / End            first / last frame
    PageUp / PageDown     jump 10 frames back / forward
    R                     restart
    F12                   screenshot

``--export-frames DIR`` renders offscreen and writes PNGs instead of opening a window: one image per replay
step (after its animations settle) or, with ``--sim-fps N``, N images per simulated second.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

log = logging.getLogger(__name__)


def _parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError as e:
        raise argparse.ArgumentTypeError("size must look like 1600x900") from e


def export_frames(replay, out_dir: str, size=(1280, 720), quality: str = "medium", pov: str = "omniscient",
                  sim_fps: float = 0.0, every: int = 1, start: int = 0, stop: int | None = None,
                  progress=None) -> list[str]:
    """Render ``replay`` offscreen and write PNG files; returns the written paths."""
    from PIL import Image
    from emberdeep.render import Renderer
    os.makedirs(out_dir, exist_ok=True)
    r = Renderer("rgb_array", size[0], size[1], quality, hud=True, audio=False, pov=pov)
    paths: list[str] = []
    try:
        r.set_scene(replay.scene)
        n = 0
        stop = len(replay.frames) if stop is None else min(stop, len(replay.frames))
        for i in range(start, stop):
            frame, events = replay.frames[i]
            r.push(frame, events)
            if sim_fps and sim_fps > 0:
                step_dt = 1.0 / 60.0
                every_n = max(1, int(round(60.0 / sim_fps)))
                k, guard = 0, 0
                while not r.director.idle and guard < 60 * 120:
                    r.step_sim(step_dt)
                    k += 1
                    guard += 1
                    if k % every_n == 0:
                        p = os.path.join(out_dir, f"frame_{n:06d}.png")
                        Image.fromarray(r.capture()).save(p)
                        paths.append(p)
                        n += 1
            else:
                r.play_until_idle()
                if (i - start) % max(1, every) == 0:
                    p = os.path.join(out_dir, f"frame_{n:06d}.png")
                    Image.fromarray(r.capture()).save(p)
                    paths.append(p)
                    n += 1
            if progress:
                progress(i + 1, stop)
        if sim_fps and sim_fps > 0:
            p = os.path.join(out_dir, f"frame_{n:06d}.png")
            Image.fromarray(r.capture()).save(p)
            paths.append(p)
    finally:
        r.close()
    return paths


class ReplayPlayer:
    """Interactive playback with scrubbing on top of a human-mode renderer."""

    def __init__(self, renderer, replay, start: int = 0):
        self.r = renderer
        self.replay = replay
        self.idx = max(0, min(start, len(replay.frames) - 1))
        self._pending_jump: int | None = None
        self._install_keys()

    # ------------------------------------------------------------------------------------ keys
    def _install_keys(self) -> None:
        import pygame as pg
        from emberdeep.render.hud_api import UICommand
        hud = self.r.hud
        orig = hud.handle_event

        def handle(ev, view):
            if ev.type == pg.KEYDOWN:
                n = len(self.replay.frames)
                if ev.key == pg.K_RIGHT:
                    self._pending_jump = min(n - 1, self.idx + 1)
                elif ev.key == pg.K_LEFT:
                    self._pending_jump = max(0, self.idx - 1)
                elif ev.key == pg.K_HOME:
                    self._pending_jump = 0
                elif ev.key == pg.K_END:
                    self._pending_jump = n - 1
                elif ev.key == pg.K_PAGEUP:
                    self._pending_jump = max(0, self.idx - 10)
                elif ev.key == pg.K_PAGEDOWN:
                    self._pending_jump = min(n - 1, self.idx + 10)
                elif ev.key == pg.K_r:
                    self._pending_jump = 0
                    self.r.paused = False
                if self._pending_jump is not None:
                    return UICommand("noop")
            return orig(ev, view)

        hud.handle_event = handle

    # ------------------------------------------------------------------------------------ control
    def _jump(self, i: int) -> None:
        d = self.r.director
        d.queue.clear()
        d.cur = None
        d.g = None
        d.skipping = False
        self.r.particles.clear()
        frame, _events = self.replay.frames[i]
        d.push(frame, [])
        self.idx = i

    def run(self) -> None:
        r = self.r
        r.set_scene(self.replay.scene)
        frames = self.replay.frames
        frame, events = frames[self.idx]
        r.push(frame, events if self.idx == 0 else [])
        while r.is_open():
            if self._pending_jump is not None:
                self._jump(self._pending_jump)
                self._pending_jump = None
            elif not r.paused and r.director.idle and self.idx + 1 < len(frames):
                self.idx += 1
                f, ev = frames[self.idx]
                r.push(f, ev)
            r.run_frame()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m emberdeep.viewer", description="Play or export an Emberdeep replay.")
    ap.add_argument("file", help=".edr replay file")
    ap.add_argument("--pov", default="omniscient", help="omniscient | <agent id> | faction:<n>")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--size", type=_parse_size, default=(1600, 900), help="window / image size, e.g. 1600x900")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--no-sound", action="store_true", help="disable procedural audio")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--export-frames", metavar="DIR", help="write PNG frames to DIR (offscreen) instead of playing")
    ap.add_argument("--sim-fps", type=float, default=0.0, help="with --export-frames: images per simulated second")
    ap.add_argument("--every", type=int, default=1, help="with --export-frames: keep every N-th step")
    ap.add_argument("--info", action="store_true", help="print replay metadata and exit")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    from emberdeep.replay import load_replay
    replay = load_replay(a.file)
    if a.info:
        m = replay.meta
        print(f"{a.file}: scenario={m.get('scenario')} seed={m.get('seed')} steps={len(replay.frames) - 1} "
              f"agents={m.get('agents')} outcome={m.get('outcome', {}).get('reason')}")
        return 0
    if a.export_frames:
        def prog(i, n):
            if i % 20 == 0 or i == n:
                print(f"\rrendered {i}/{n}", end="", file=sys.stderr, flush=True)
        paths = export_frames(replay, a.export_frames, a.size, a.quality, a.pov, a.sim_fps, a.every, a.start,
                              progress=prog)
        print(f"\nwrote {len(paths)} images to {a.export_frames}")
        return 0
    from emberdeep.render import Renderer
    r = Renderer("human", a.size[0], a.size[1], a.quality, hud=not a.no_hud, audio=not a.no_sound, pov=a.pov)
    r.speed = a.speed
    try:
        ReplayPlayer(r, replay, a.start).run()
    finally:
        r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
