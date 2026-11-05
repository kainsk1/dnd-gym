"""Standalone HUD demo (software pygame, no OpenGL).

Interactive::

    python -m emberdeep.render.ui.demo [--size 1600x900] [--state human]

Headless screenshot::

    python -m emberdeep.render.ui.demo --screenshot docs/screenshots/hud_combat.png --state combat
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

STATES = ("combat", "human", "human_aoe", "inspector", "scoreboard", "tooltip", "exploration", "banner", "research",
          "help", "reaction")


class Demo:
    def __init__(self, size: tuple[int, int], headless: bool = False, audio=None) -> None:
        import pygame
        from emberdeep.render.ui import _mock_frames as M
        from emberdeep.render.ui.hud import HUD
        self.pg, self.M = pygame, M
        self.size = size
        self.headless = headless
        pygame.init()
        self.screen = pygame.display.set_mode(size) if not headless else pygame.display.set_mode(size)
        pygame.display.set_caption("Emberdeep HUD demo")
        self.fight = M.MockFight()
        self.scene = self.fight.scene
        self.hud = HUD(audio=audio)
        self.hud.set_scene(self.scene)
        self.overlay = pygame.Surface(size, pygame.SRCALPHA)
        self.idx = -1
        self.frame = self.fight.steps[0][0]
        self.paused = headless
        self.speed = 1.0
        self.pov = "omniscient"
        self.selected = -1
        self.hovered_eid = -1
        self.hovered_cell = None
        self.options: dict = {}
        self.human = False
        self.time = 0.0
        self.step_timer = 0.0
        self.queue: list[tuple[float, dict]] = []
        self.audio = audio
        self.quit = False
        self.shot_requested = False
        self._bg_key = None
        self._bg = None
        self.pending_shots: list[str] = []

    # ------------------------------------------------------------- view
    def view(self):
        v = self.M.make_view(self.size, self.frame, self.scene, self.pov, self.hovered_eid, self.selected, self.human,
                             self.options)
        v.paused, v.speed, v.time = self.paused, self.speed, self.time
        v.hovered_cell = self.hovered_cell
        return v

    # ------------------------------------------------------------- stepping
    def advance(self) -> bool:
        if self.idx + 1 >= len(self.fight.steps):
            return False
        self.idx += 1
        fr, evs = self.fight.steps[self.idx]
        self.frame = fr
        self.hud.on_frame(fr)
        t0 = 0.0
        for e in evs:
            self.queue.append((self.time + t0, e))
            t0 += 0.35
        return True

    def pump_events(self) -> None:
        v = self.view()
        while self.queue and self.queue[0][0] <= self.time:
            _, e = self.queue.pop(0)
            self.hud.on_event(e, self.frame, v)
            if self.audio is not None:
                try:
                    self.audio.on_event(e)
                except Exception:  # noqa: BLE001
                    pass

    def tick(self, dt: float) -> None:
        v = self.view()
        sdt = dt * (0.0 if self.paused and not self.queue else self.speed)
        self.time += dt * self.speed if not self.paused else dt * self.speed * 0.0
        self.hud.update(dt, v)
        self.pump_events()
        if not self.paused and self.idx + 1 < len(self.fight.steps):
            self.step_timer += dt * self.speed
            if self.step_timer > 2.6 and not self.queue:
                self.step_timer = 0
                self.advance()
        if self.audio is not None:
            try:
                self.audio.update(dt)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------- drawing
    def background(self):
        pg = self.pg
        key = self.size
        if self._bg_key == key:
            return self._bg
        W, H = self.size
        bg = pg.Surface((W, H))
        bg.fill((10, 10, 14))
        v = self.view()
        cp = v.cell_px
        rng = np.random.default_rng(1)
        cols = {1: (46, 42, 40), 2: (22, 21, 26), 4: (30, 52, 78), 8: (60, 56, 52), 12: (56, 90, 60)}
        for y in range(self.scene.height):
            for x in range(self.scene.width):
                t = int(self.scene.terrain[y, x])
                base = cols.get(t, (40, 38, 38))
                j = int(rng.integers(-5, 6))
                c = tuple(max(0, min(255, b + j)) for b in base)
                x0, y0 = v.world_to_screen(x, y)
                pg.draw.rect(bg, c, (x0, y0, cp + 1, cp + 1))
                if t == 1:
                    pg.draw.rect(bg, (34, 31, 30), (x0, y0, cp + 1, cp + 1), 1)
        self._bg, self._bg_key = bg, key
        return bg

    def render(self) -> "pygame.Surface":
        pg = self.pg
        v = self.view()
        self.screen.blit(self.background(), (0, 0))
        fr = self.frame
        for e in fr.entities:
            cx, cy, r = v.entity_screen[e.id]
            col = self.hud.faction_color(e.faction) if e.faction >= 0 else (160, 140, 90)
            if e.dead:
                pg.draw.circle(self.screen, (60, 30, 30), (cx, cy), r * 0.6)
                continue
            pg.draw.circle(self.screen, (0, 0, 0), (cx + 3, cy + 4), r)
            pg.draw.circle(self.screen, tuple(int(c * 0.55) for c in col), (cx, cy), r)
            pg.draw.circle(self.screen, col, (cx, cy), r, max(2, int(r * 0.12)))
            ic = self.hud.ui.icon(e.icon, int(r * 1.3), (235, 228, 210))
            self.screen.blit(ic, ic.get_rect(center=(cx, cy)))
            if e.id == v.selected_eid or e.id == v.hovered_eid:
                pg.draw.circle(self.screen, (255, 235, 160), (cx, cy), r + 5, 2)
        self.overlay.fill((0, 0, 0, 0))
        self.hud.draw(self.overlay, v)
        self.screen.blit(self.overlay, (0, 0))
        return self.screen

    # ------------------------------------------------------------- commands
    def apply(self, cmd) -> None:
        if cmd is None:
            return
        k, val = cmd.kind, cmd.value
        if k == "toggle_pause":
            self.paused = not self.paused
        elif k == "pause":
            self.paused = True
        elif k == "resume":
            self.paused = False
        elif k == "step":
            self.advance()
        elif k == "speed":
            self.speed = float(val)
        elif k == "pov":
            self.pov = val
        elif k == "select":
            self.selected = int(val)
        elif k == "toggle":
            self.options[val] = not self.hud.opt(self.view(), val)
        elif k == "choose_action":
            print(f"[demo] choose_action {val}")  # noqa: T201 (demo only)
            self.hud.end_human_choice()
            self.human = False
        elif k == "skip_anim":
            self.time += 5
            self.pump_events()
        elif k == "quit":
            self.quit = True
        elif k == "screenshot":
            self.pending_shots.append(f"hud_demo_{int(time.time())}.png")

    def handle(self, ev) -> None:
        pg = self.pg
        if ev.type == pg.QUIT:
            self.quit = True
            return
        if ev.type == pg.MOUSEMOTION:
            self._hover(ev.pos)
        v = self.view()
        cmd = self.hud.handle_event(ev, v)
        if cmd is None and ev.type == pg.MOUSEBUTTONDOWN and ev.button in (1, 3):
            cell = self._cell_at(ev.pos)
            cmd = self.hud.world_click(cell, self.hovered_eid, ev.button, v)
        self.apply(cmd)

    def _cell_at(self, pos):
        v = self.view()
        x, y = v.screen_to_world(*pos)
        return (int(math.floor(x)), int(math.floor(y)))

    def _hover(self, pos) -> None:
        v = self.view()
        self.hovered_eid = -1
        for eid, (cx, cy, r) in v.entity_screen.items():
            if (cx - pos[0]) ** 2 + (cy - pos[1]) ** 2 <= r * r:
                self.hovered_eid = eid
        self.hovered_cell = self._cell_at(pos)

    def mouse_to(self, pos) -> None:
        pg = self.pg
        ev = pg.event.Event(pg.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0))
        self.handle(ev)

    def start_human(self, agent: str = "rogue") -> None:
        self.human = True
        self.hud.begin_human_choice(agent, self.M.make_actions())

    # ------------------------------------------------------------- main loops
    def run(self) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        self.advance()
        self.paused = False
        while not self.quit:
            dt = clock.tick(60) / 1000.0
            for ev in pg.event.get():
                self.handle(ev)
                if ev.type == pg.KEYDOWN and ev.key == pg.K_F9:
                    self.start_human()
            self.tick(dt)
            self.render()
            pg.display.flip()
        pg.quit()

    def simulate(self, seconds: float, fps: int = 30) -> None:
        n = int(seconds * fps)
        for _ in range(n):
            self.time += 1.0 / fps
            self.hud.update(1.0 / fps, self.view())
            self.pump_events()
            self.render()

    def play_to(self, last_step: int, gap: float = 1.6) -> None:
        while self.idx < last_step:
            self.advance()
            self.simulate(gap if self.idx < last_step else 0.0)
        # flush
        self.simulate(0.05)


def run_state(demo: Demo, state: str) -> None:
    M = demo.M
    if state == "exploration":
        demo.frame = M.make_frame("exploration")
        demo.hud.on_frame(demo.frame)
        demo.hud.on_event(M.ev(M.ET.EPISODE_START), demo.frame, demo.view())
        demo.hud.on_event(M.ev(M.ET.SAY, 3, kind=4, rnd=0), demo.frame, demo.view())
        demo.hud.on_event(M.ev(M.ET.SEARCH, 3, roll=14, total=19, found=[10], rnd=0), demo.frame, demo.view())
        demo.hud.on_event(M.ev(M.ET.CHEST_OPEN, 3, 10, rnd=0), demo.frame, demo.view())
        demo.simulate(0.8)
        return
    if state == "combat":
        demo.play_to(6)
        demo.simulate(0.8)
        return
    demo.play_to(7)
    if state != "banner":
        demo.simulate(3.4)
    if state == "banner":
        demo.hud.on_event(M.ev(M.ET.COMBAT_START, order=[3, 5], rolls={}, surprised=[]), demo.frame, demo.view())
        demo.simulate(0.9)
    elif state in ("human", "human_aoe"):
        demo.simulate(2.5)
        demo.start_human("rogue")
        demo.simulate(0.5)
        v = demo.view()
        if state == "human":
            g = next(g for g in demo.hud.actionbar.groups if g.label == "Shortsword")
            demo.hud.actionbar.sel = g
            cx, cy, r = v.entity_screen[5]
            demo.mouse_to((int(cx), int(cy)))
        else:
            g = next(g for g in demo.hud.actionbar.groups if g.label == "Fireball")
            demo.hud.actionbar.sel = g
            demo.hovered_cell = (15, 10)
            demo.hovered_eid = -1
            x, y = v.world_to_screen(15.5, 10.5)
            demo.mouse_to((int(x), int(y)))
        demo.simulate(0.6)
    elif state == "reaction":
        demo.simulate(2.5)
        demo.frame.decision = {"eid": 3, "kind": 3, "options": ["Uncanny Dodge", "Take the hit"]}
        A = M.ActionDescriptor
        demo.human = True
        demo.hud.begin_human_choice("rogue", [A(40, "decision", "Uncanny Dodge (halve the damage)", "uncanny_dodge", "shield-bounces", "reaction"),
                                              A(41, "decision", "Take the hit", "", "cancel", "free")])
        demo.simulate(0.8)
    elif state == "inspector":
        demo.simulate(2.5)
        demo.selected = 4
        demo.simulate(0.4)
    elif state == "tooltip":
        demo.simulate(2.5)
        demo.pov = "fighter"
        v = demo.view()
        cx, cy, r = v.entity_screen[8]
        demo.mouse_to((int(cx), int(cy)))
        demo.simulate(0.8)
    elif state == "research":
        demo.options["research"] = True
        demo.simulate(2.5)
    elif state == "help":
        demo.simulate(1.5)
        demo.hud.help_open = True
        demo.simulate(0.3)
    elif state == "scoreboard":
        demo.frame.done = True
        demo.frame.outcome = "The Company drove the goblins from the warren and claimed their hoard."
        demo.hud.on_frame(demo.frame)
        demo.hud.on_event(M.ev(M.ET.EPISODE_END, reason="Warren cleared", winners=[0]), demo.frame, demo.view())
        demo.simulate(3.8)
        demo.simulate(0.6)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--state", default="combat", choices=STATES)
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--bench", action="store_true", help="print HUD draw time statistics")
    a = ap.parse_args(argv)
    w, h = (int(x) for x in a.size.lower().split("x"))
    if a.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    audio = None
    if not a.no_audio and not a.screenshot:
        from emberdeep.render.audio import AudioEngine
        audio = AudioEngine()
    demo = Demo((w, h), headless=bool(a.screenshot), audio=audio)
    if a.screenshot:
        run_state(demo, a.state)
        demo.render()
        os.makedirs(os.path.dirname(os.path.abspath(a.screenshot)), exist_ok=True)
        demo.pg.image.save(demo.screen, a.screenshot)
        if a.bench:
            ts = []
            for _ in range(60):
                t0 = time.perf_counter()
                demo.overlay.fill((0, 0, 0, 0))
                demo.hud.draw(demo.overlay, demo.view())
                ts.append((time.perf_counter() - t0) * 1000)
            print(f"HUD draw: mean {np.mean(ts):.2f} ms  p95 {np.percentile(ts, 95):.2f} ms  max {max(ts):.2f} ms")  # noqa: T201
        return 0
    demo.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
