"""The HUD: orchestrates every 2D component and implements ``HUDProtocol``.

Immediate-mode design: every frame ``draw`` lays the panels out for the current screen size,
draws them (all heavy art is cached) and rebuilds a list of hotspots (:class:`Hot`) used by
``handle_event`` for clicks, wheel, drag and by the tooltip layer for hovers.
"""
from __future__ import annotations

import logging
import time
from collections import deque

import pygame

from emberdeep.core.events import EventType as ET
from emberdeep.render.api import ActionDescriptor, EntityView, FrameState, StaticScene
from emberdeep.render.hud_api import HUDProtocol, UICommand, ViewInfo
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.actionbar import ActionBar
from emberdeep.render.ui.banners import Banners
from emberdeep.render.ui.combat_log import CombatLog, humanize
from emberdeep.render.ui.controls import SPEEDS, Controls
from emberdeep.render.ui.floating import FloatingLayer
from emberdeep.render.ui.initiative import Ribbon
from emberdeep.render.ui.inspector import Inspector
from emberdeep.render.ui.minimap import Minimap
from emberdeep.render.ui.party import PartyFrames
from emberdeep.render.ui.research import ResearchPanel
from emberdeep.render.ui.scoreboard import Scoreboard, draw_help
from emberdeep.render.ui.tooltips import TipLayer, cond_tip, CONDITIONS
from emberdeep.render.ui.widgets import Hot, Tip

log = logging.getLogger(__name__)

HP_BUCKETS = ((1.0, "Unhurt"), (0.75, "Scratched"), (0.5, "Wounded"), (0.25, "Bloodied"), (0.0, "Near death"))


class HUD(HUDProtocol):
    DEFAULT_OPTIONS = {"grid": False, "threat": False, "los": False, "labels": False, "log": True,
                       "minimap": True, "research": False, "mute": False}

    def __init__(self, audio=None) -> None:
        self.ui = T.UI()
        self.audio = audio
        self.scene: StaticScene | None = None
        self.frame: FrameState | None = None
        self.view: ViewInfo | None = None
        self.t = 0.0
        self.dt = 1 / 60
        self.speed = 1.0
        self.mouse = (-1, -1)
        self._saw_motion = False
        self.view_human_turn = False
        self.view_selected = -1
        self.hots: list[Hot] = []
        self._new_hots: list[Hot] = []
        self._capture: list[Hot] | None = None
        self.safe: pygame.Rect | None = None
        # --- change tracking: ``dirty`` is valid after update(); False means the last draw is still exact
        self.dirty = True
        self._force = True
        self._sig = None
        self._ps = -1
        self._mouse_t = -9.0
        self.version = 0
        self._P: pygame.Surface | None = None
        self._P_scale = 0.0
        self._pc: dict = {}
        self._seen: set = set()
        self.panel_redraws = 0
        self.modal_prev = False
        self._modal_now = False
        self._in_modal = False
        self._drag: Hot | None = None
        self.help_open = False
        self.final_kind: str | None = None
        self._emap: dict[int, EntityView] = {}
        self._names: dict[int, tuple[str, int]] = {}
        self._pov_faction_cache: tuple | None = None
        self.history: dict[str, deque] = {}
        self.social: dict[tuple[int, int], dict[str, float]] = {}
        self._last_click: tuple[int, float] = (-1, 0.0)
        self.draw_ms = 0.0
        # components
        self.log = CombatLog(self)
        self.ribbon = Ribbon(self)
        self.party = PartyFrames(self)
        self.floating = FloatingLayer(self)
        self.banners = Banners(self)
        self.tips = TipLayer(self)
        self.inspector = Inspector(self)
        self.minimap = Minimap(self)
        self.controls = Controls(self)
        self.actionbar = ActionBar(self)
        self.research = ResearchPanel(self)
        self.scoreboard = Scoreboard(self)

    # ================================================================= protocol
    def set_scene(self, scene: StaticScene) -> None:
        self._force = True
        self.scene = scene
        self.frame = None
        self._emap.clear()
        self.history.clear()
        self.social.clear()
        self.log.clear()
        self.banners.clear()
        self.floating.floaters.clear()
        self.floating.bubbles.clear()
        self.scoreboard.reset()
        self.final_kind = None
        self.minimap.set_scene(scene)
        self.actionbar.end_choice()

    def on_frame(self, frame: FrameState) -> None:
        self._force = True
        self._ingest(frame)
        self.party.on_frame(frame)
        for ag, ret in frame.returns.items():
            self.history.setdefault(ag, deque(maxlen=160)).append(float(ret))
        if frame.done and self.final_kind is None:
            self.final_kind = self.outcome_kind({}, frame)
            if self.banners.final is None:
                from emberdeep.render.ui.banners import Banner
                names = {"victory": ("VICTORY", T.GOLD_HI, (130, 90, 10)), "defeat": ("DEFEAT", T.RED, (120, 16, 12))}
                title, col, glow = names.get(self.final_kind, ("EPISODE OVER", T.TEXT, (60, 60, 70)))
                self.banners.push(Banner(title, "", col, glow, "final", 99, pt=72))
        elif not frame.done:
            self.final_kind = None

    def on_event(self, event: dict, frame: FrameState, view: ViewInfo) -> None:
        self._force = True
        try:
            if frame is not None:
                if self.frame is None or frame.step >= self.frame.step:
                    pass
                for e in frame.entities:
                    self._names[e.id] = (e.name, e.faction)
                    self._emap.setdefault(e.id, e)
            self.log.add(event)
            self.floating.on_event(event, view)
            self.banners.on_event(event, frame or self.frame)
            self.ribbon.on_event(event)
            self._social(event)
            if event.get("t") == ET.EPISODE_END:
                self.final_kind = self.outcome_kind(event, frame or self.frame)
        except Exception:  # noqa: BLE001
            log.exception("HUD.on_event failed for %s", event)

    def update(self, dt: float, view: ViewInfo) -> None:
        dt = min(max(dt, 0.0), 0.25)
        self.dt = dt
        self.t += dt
        self.view = view
        self.speed = view.speed
        self.view_human_turn = view.human_turn
        self.view_selected = view.selected_eid
        self.ui.set_size(*view.screen_size)
        if not self._saw_motion:
            try:
                if pygame.mouse.get_focused():
                    self.mouse = pygame.mouse.get_pos()
            except Exception:  # noqa: BLE001
                pass
        self.party.update(dt)
        self.floating.update(dt)
        self.banners.update(dt)
        self.scoreboard.update(dt)
        self.dirty = self._compute_dirty(view)

    def _compute_dirty(self, view: ViewInfo) -> bool:
        """True when anything visible may differ from the last draw (inputs, events, animation, camera)."""
        d = self._force or self.frame is None
        try:
            es = tuple((k, int(v[0]), int(v[1]), int(v[2])) for k, v in view.entity_screen.items())
            x0, y0 = view.world_to_screen(0, 0)
            sig = (view.screen_size, es, view.hovered_eid, view.hovered_cell, view.selected_eid, view.pov, view.paused,
                   view.speed, view.human_turn, tuple(sorted(view.options.items())), int(x0), int(y0),
                   round(view.cell_px, 1), tuple(self.mouse))
        except Exception:  # noqa: BLE001
            sig = object()
        if sig != self._sig:
            if self._sig is not None and sig[-1] != self._sig[-1]:
                self._mouse_t = self.t
            self._sig = sig
            d = True
        if self.t - self._mouse_t < 0.7:                     # tooltip delay / fade
            d = True
        fr = self.frame
        if fr is not None and not d:
            fl = self.floating
            if fl.floaters or fl.bubbles or self.banners.big or self.banners.small or self.banners.final is not None:
                d = True
            elif self.party.rflash or self.ribbon.moving or self.scoreboard.shown or self.help_open or fr.done:
                d = True
            elif any(abs(a.shown - a.last) > 0.01 or a.ghost - a.shown > 0.01 or a.flash > 0 or a.hurt > 0
                     for a in self.party.anim.values()):
                d = True
            elif self.log.entries and self.t - self.log.entries[-1].born < 1.4:
                d = True
            elif fr.mode == "combat" or self.actionbar.active:   # pulsing highlights: redraw at 10 Hz
                ps = int(self.t * 10)
                if ps != self._ps:
                    d = True
        if d:
            self._ps = int(self.t * 10)
        return d

    def draw(self, surface: pygame.Surface, view: ViewInfo) -> None:
        t0 = time.perf_counter()
        ui = self.ui
        W, H = surface.get_size()
        ui.set_size(W, H)
        self.view = view
        self._new_hots = []
        self._modal_now = False
        self._in_modal = False
        if self.frame is None or self.scene is None:
            self.hots, self._new_hots = [], []
            return
        try:
            self._draw(surface, view, W, H)
        except Exception:  # noqa: BLE001
            log.exception("HUD.draw failed")
        self.hots = self._new_hots
        self.modal_prev = self._modal_now
        self.dirty = False
        self._force = False
        self.version += 1
        self.draw_ms = (time.perf_counter() - t0) * 1000.0

    # ---- per-panel cached composition -------------------------------------------------------------
    def _panel(self, name: str, surf: pygame.Surface, rect: pygame.Rect, key, fn, extra: pygame.Rect | None = None) -> None:
        """Render a panel into the shared panel layer only when ``key`` changed, then copy its region to ``surf``."""
        ui = self.ui
        P = self._P
        mg = ui.s(7)
        region = rect.inflate(2 * mg, 2 * mg)
        if extra is not None:
            region = region.union(extra)
        region = region.clip(P.get_rect())
        mp = self.mouse
        if region.collidepoint(mp):
            key = (key, mp)
        ent = self._pc.get(name)
        if ent is None or ent[0] != key or ent[1] != region:
            if ent is not None and ent[1] != region:
                P.fill((0, 0, 0, 0), ent[1])
            P.set_clip(region)
            P.fill((0, 0, 0, 0), region)
            prev = self._capture
            self._capture = []
            try:
                fn(P)
                hots = self._capture
            finally:
                self._capture = prev
                P.set_clip(None)
            ent = (key, region, hots)
            self._pc[name] = ent
            self.panel_redraws += 1
        self._seen.add(name)
        surf.blit(self._P, region.topleft, region, special_flags=pygame.BLEND_RGBA_ADD)
        self._new_hots.extend(ent[2])

    def _draw(self, surf: pygame.Surface, view: ViewInfo, W: int, H: int) -> None:
        ui = self.ui
        fr = self.frame
        if self._P is None or self._P.get_size() != (W, H) or self._P_scale != ui.scale:
            self._P = pygame.Surface((W, H), pygame.SRCALPHA)
            self._P_scale = ui.scale
            self._pc.clear()
        self._seen = set()
        m = ui.s(14)
        gap = ui.s(10)
        colw = ui.s(344)
        party_w = ui.s(292)
        step = int(self.t * 10)
        combat = fr.mode == "combat"
        fid = id(fr)
        # ---- left: party
        prect = pygame.Rect(m, m, party_w, H - 2 * m)
        pkey = (fid, self.party.compact, fr.active_id, view.selected_eid, view.pov,
                tuple((e.id, int(a.shown * 4), int(a.ghost * 4), int(a.flash * 20), int(a.hurt * 20))
                      for e in self.party.members() for a in [self.party.anim.get(e.id)] if a is not None),
                tuple((k, int(v[1] * 10)) for k, v in self.party.rflash.items()))
        self._panel("party", surf, prect, pkey, lambda s_: self.party.draw(s_, prect, view))
        # ---- top centre: ribbon
        left_edge = m + party_w + gap
        right_edge = W - m - colw - gap
        rb = pygame.Rect(left_edge, m, max(ui.s(300), right_edge - left_edge), ui.s(124))
        rkey = (fid, fr.active_id, view.pov, self.ribbon.moving,
                self.t if self.ribbon.moving else 0, tuple(sorted(self.ribbon.surprised)))
        self._panel("ribbon", surf, rb, rkey, lambda s_: self.ribbon.draw(s_, pygame.Rect(rb.x, rb.y, rb.w, 0), view))
        self.safe = pygame.Rect(left_edge - gap, 0, right_edge - left_edge + gap, H)
        # ---- right column
        rx = W - m - colw
        y = m
        if self.opt(view, "minimap"):
            sc = self.scene
            aspect = sc.width / max(1, sc.height)
            pad = ui.s(10)
            map_h = min(ui.s(168), int((colw - 2 * pad) / aspect))
            mh = pad * 2 + ui.s(22) + map_h
            mr = pygame.Rect(rx, y, colw, mh)
            x0, y0 = view.screen_to_world(0, 0)
            x1, y1 = view.screen_to_world(*view.screen_size)
            mkey = (fid, view.pov, fr.active_id, int(x0 * 6), int(y0 * 6), int(x1 * 6), int(y1 * 6))
            self._panel("minimap", surf, mr, mkey, lambda s_: self.minimap.draw(s_, mr, view))
            y += mh + gap
        ctl_h = ui.s(30) + ui.s(8) + ui.s(24) + ui.s(8) + ui.s(26) + ui.s(20)
        ctl_rect = pygame.Rect(rx, H - m - ctl_h, colw, ctl_h)
        if self.opt(view, "research"):
            rh = min(self.research.height(), max(ui.s(120), ctl_rect.y - y - gap - ui.s(120)))
            rr = pygame.Rect(rx, y, colw, rh)
            self._panel("research", surf, rr, (fid, view.pov, fr.active_id, self.history and len(next(iter(self.history.values())))),
                        lambda s_: self.research.draw(s_, rr, view))
            y += rh + gap
        if self.opt(view, "log"):
            lh = ctl_rect.y - gap - y
            if self.log.collapsed:
                lh = ui.s(40)
            if lh > ui.s(60):
                lr = pygame.Rect(rx, y, colw, lh)
                lg = self.log
                last = lg.entries[-1] if lg.entries else None
                age = self.t - last.born if last is not None else 9.0
                lkey = (len(lg.entries), id(last), lg.filter, lg.collapsed, int(lg.scroll), lg._unseen,
                        int(age * 8) if age < 1.4 else -1)
                self._panel("log", surf, lr, lkey, lambda s_: lg.draw(s_, lr))
        pov_extra = None
        if self.controls.pov_open:
            n_opt = len(self.pov_options(view))
            pov_extra = pygame.Rect(ctl_rect.x, ctl_rect.y - ui.s(28) * n_opt - ui.s(30), colw, ui.s(28) * n_opt + ui.s(30))
        ckey = (view.paused, view.speed, view.pov, tuple(sorted(view.options.items())), self.controls.pov_open,
                self.modal_prev)
        self._panel("controls", surf, ctl_rect, ckey, lambda s_: self.controls.draw(s_, ctl_rect, view), pov_extra)
        # ---- inspector (docked left of the right column)
        if self.inspector.visible(view):
            iw = ui.s(336)
            ix = rx - gap - iw
            iy = self.ribbon.rect.bottom + gap if self.ribbon.rect.h else m
            ih = H - m - iy - (self.actionbar.rect.h + gap if self.actionbar.active and self.actionbar.rect.right > ix else 0)
            ir = pygame.Rect(ix, iy, iw, max(ui.s(200), ih))
            self._panel("inspector", surf, ir, (fid, view.selected_eid, view.pov, int(self.inspector.scroll), ir),
                        lambda s_: self.inspector.draw(s_, ir, view))
        for k in list(self._pc):
            if k not in self._seen:                      # panel disappeared: wipe its region
                self._P.fill((0, 0, 0, 0), self._pc[k][1])
                del self._pc[k]
        # ---- world-space overlays (cheap, redrawn every frame)
        self.party.draw_overlay(surf)
        self.ribbon.draw_overlay(surf)
        self.floating.draw_nameplates(surf, view)
        self.floating.draw(surf, view)
        # ---- banners
        self.banners.draw(surf, view, self.ribbon.rect.bottom if self.ribbon.rect.h else m)
        # ---- human play
        if self.actionbar.active:
            area = pygame.Rect(left_edge - gap, 0, right_edge - left_edge + gap, H - m)
            self.actionbar.draw(surf, area, view)
            if self.actionbar.decisions:
                self._modal_now = True
                self._in_modal = True
        # ---- modals
        if self.scoreboard.shown and not self.scoreboard.dismissed:
            self._modal_now = True
            self._in_modal = True
            self.scoreboard.draw(surf, view)
        if self.help_open:
            self._modal_now = True
            self._in_modal = True
            draw_help(self, surf)
        self.tips.draw(surf)

    # ================================================================= hotspots
    def hot(self, rect: pygame.Rect, tip=None, click=None, right=None, wheel=None, drag=None, block: bool = False,
            name: str = "") -> None:
        h = Hot(pygame.Rect(rect), tip, click, right, wheel, drag, block, name)
        if self._capture is not None:
            self._capture.append(h)
        else:
            self._new_hots.append(h)

    def mouse_blocked(self, rect: pygame.Rect) -> bool:
        return self.modal_prev and not self._in_modal

    def mouse_over_panel(self) -> bool:
        mp = self.mouse
        return any(h.block and h.rect.collidepoint(mp) for h in self.hots)

    def _pick(self, pos, attr: str | None = None) -> Hot | None:
        for h in reversed(self.hots):
            if not h.rect.collidepoint(pos):
                continue
            if attr is None:
                if h.block or h.click or h.right or h.wheel or h.drag:
                    return h
            elif getattr(h, attr) is not None:
                return h
            elif h.block:
                return None
        return None

    def current_tip(self) -> Tip | None:
        mp = self.mouse
        for h in reversed(self.hots):
            if not h.rect.collidepoint(mp):
                continue
            if h.tip is not None:
                try:
                    return h.tip() if callable(h.tip) else h.tip
                except Exception:  # noqa: BLE001
                    log.debug("tip failed", exc_info=True)
                    return None
            if h.block:
                return None
        v = self.view
        if v is not None and v.hovered_eid >= 0 and self.frame is not None:
            e = self.ent(v.hovered_eid)
            if e is not None and not (self.actionbar.sel is not None):
                return self.entity_tip(e, v)
        return None

    # ================================================================= input
    def handle_event(self, ev, view: ViewInfo) -> UICommand | None:  # noqa: C901
        try:
            return self._handle(ev, view)
        except Exception:  # noqa: BLE001
            log.exception("HUD.handle_event failed")
            return None

    def _noop(self) -> UICommand:
        return UICommand("noop")

    def _handle(self, ev, view: ViewInfo) -> UICommand | None:  # noqa: C901
        t = ev.type
        if t != pygame.MOUSEMOTION:
            self._force = True
        if t == pygame.MOUSEMOTION:
            self.mouse = ev.pos
            self._saw_motion = True
            if self._drag is not None and self._drag.drag is not None:
                res = self._drag.drag(ev.pos)
                return res if isinstance(res, UICommand) else self._noop()
            return None
        if t == pygame.MOUSEBUTTONDOWN:
            pos = ev.pos
            self.mouse = pos
            self._saw_motion = True
            if ev.button in (4, 5):
                return self._noop() if self._pick(pos) is not None else None
            was_open = self.controls.pov_open
            hot = self._pick(pos)
            res = None
            if hot is not None:
                if ev.button == 1:
                    if hot.drag is not None:
                        self._drag = hot
                    cb = hot.click
                elif ev.button == 3:
                    cb = hot.right
                else:
                    cb = None
                if cb is not None:
                    res = cb()
                    if self.audio is not None and ev.button == 1:
                        try:
                            self.audio.on_ui("click")
                        except Exception:  # noqa: BLE001
                            pass
            if was_open and self.controls.pov_open:
                self.controls.pov_open = False
            if self.help_open and hot is not None and hot.block:
                pass
            if isinstance(res, UICommand):
                return res
            if hot is not None:
                return self._noop()
            return None
        if t == pygame.MOUSEBUTTONUP:
            d, self._drag = self._drag, None
            return self._noop() if d is not None else None
        if t == pygame.MOUSEWHEEL:
            h = self._pick(self.mouse, "wheel")
            if h is not None:
                h.wheel(int(getattr(ev, "y", 0)))
                return self._noop()
            return self._noop() if self.mouse_over_panel() else None
        if t == pygame.KEYDOWN:
            return self._key(ev, view)
        return None

    def toggle_help(self):
        self.help_open = not self.help_open

    def _key(self, ev, view: ViewInfo) -> UICommand | None:  # noqa: C901
        k = ev.key
        mods = getattr(ev, "mod", 0)
        if k == pygame.K_F1:
            self.help_open = not self.help_open
            return self._noop()
        if self.help_open:
            if k == pygame.K_ESCAPE:
                self.help_open = False
            return self._noop()
        if self.scoreboard.shown and k in (pygame.K_ESCAPE, pygame.K_RETURN):
            self.scoreboard._dismiss()
            return self._noop()
        if self.actionbar.active:
            cmd = self.actionbar.key(ev)
            if cmd is not None:
                return cmd
            if self.actionbar.decisions:
                return self._noop()
        if k == pygame.K_ESCAPE:
            if self.controls.pov_open:
                self.controls.pov_open = False
                return self._noop()
            if self.inspector.visible(view):
                self.inspector.closed_for = view.selected_eid
                return UICommand("select", -1)
            return None
        if (mods & pygame.KMOD_CTRL) and k == pygame.K_q:
            return UICommand("quit")
        if mods & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META):
            return None
        if k == pygame.K_SPACE:
            return UICommand("toggle_pause")
        if k in (pygame.K_n, pygame.K_PERIOD):
            return UICommand("step")
        if k == pygame.K_TAB:
            return UICommand("skip_anim")
        if k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            i = min(range(len(SPEEDS)), key=lambda j: abs(SPEEDS[j] - view.speed))
            i = max(0, min(len(SPEEDS) - 1, i + (1 if k == pygame.K_RIGHTBRACKET else -1)))
            return UICommand("speed", SPEEDS[i])
        if k == pygame.K_v:
            opts = self.pov_options(view)
            if opts:
                i = opts.index(view.pov) if view.pov in opts else -1
                return UICommand("pov", opts[(i + 1) % len(opts)])
            return None
        toggles = {pygame.K_g: "grid", pygame.K_h: "threat", pygame.K_l: "los", pygame.K_b: "labels",
                   pygame.K_m: "minimap", pygame.K_F2: "research", pygame.K_F4: "mute"}
        if k in toggles:
            return self.controls._toggle(toggles[k])
        if k == pygame.K_c:
            return UICommand("toggle", "log")
        if k == pygame.K_F12:
            return UICommand("screenshot")
        if k == pygame.K_HOME and self.frame is not None and self.frame.active_id >= 0:
            return UICommand("focus", self.frame.active_id)
        return None

    def world_click(self, cell, eid: int, button: int, view: ViewInfo) -> UICommand | None:
        try:
            if self.actionbar.active:
                cmd = self.actionbar.world_click(cell, eid, button, view)
                if cmd is not None:
                    return cmd
                if self.actionbar.sel is not None:
                    return self._noop()
            if button == 1:
                now = time.monotonic()
                if eid >= 0:
                    last_e, last_t = self._last_click
                    self._last_click = (eid, now)
                    if last_e == eid and now - last_t < 0.4:
                        return UICommand("focus", eid)
                    return UICommand("select", eid)
                self._last_click = (-1, now)
                if view.selected_eid >= 0:
                    return UICommand("select", -1)
            return None
        except Exception:  # noqa: BLE001
            log.exception("HUD.world_click failed")
            return None

    def begin_human_choice(self, agent_id: str, actions: list[ActionDescriptor]) -> None:
        self._force = True
        self.actionbar.begin(agent_id, actions)

    def end_human_choice(self) -> None:
        self._force = True
        self.actionbar.end_choice()

    # ================================================================= helpers used by components
    def _ingest(self, frame: FrameState) -> None:
        self.frame = frame
        self._emap = {e.id: e for e in frame.entities}
        for e in frame.entities:
            self._names[e.id] = (e.name, e.faction)
        self._pov_faction_cache = None

    def ent(self, eid: int) -> EntityView | None:
        return self._emap.get(eid)

    def faction_color(self, idx: int) -> T.Color:
        if self.scene:
            for f in self.scene.factions:
                if f.index == idx:
                    return tuple(f.color)  # type: ignore[return-value]
        return (150, 150, 156)

    def faction_name(self, idx: int) -> str:
        if self.scene:
            for f in self.scene.factions:
                if f.index == idx:
                    return f.name
        return "Neutral" if idx < 0 else f"Faction {idx}"

    def name_token(self, eid: int) -> tuple:
        if eid is None or eid < 0:
            return ("t", "Something", T.TEXT_DIM, "body")
        nm, fac = self._names.get(eid, (f"#{eid}", -1))
        return ("t", nm, T.mix(self.faction_color(fac), (255, 255, 255), 0.18), "body_bold")

    def is_player_side(self, e: EntityView) -> bool:
        if self.scene:
            for f in self.scene.factions:
                if f.index == e.faction:
                    return f.is_player_side
        return False

    def opt(self, view: ViewInfo, key: str) -> bool:
        return bool(view.options.get(key, self.DEFAULT_OPTIONS.get(key, False)))

    def pov_options(self, view: ViewInfo) -> list[str]:
        if view.pov_options:
            return list(view.pov_options)
        out = ["omniscient"]
        if self.scene:
            out += list(self.scene.agent_ids) + [f"faction:{f.index}" for f in self.scene.factions]
        return out

    def _pov_faction(self, view: ViewInfo) -> int | None:
        pov = view.pov
        if pov == "omniscient" or self.frame is None:
            return None
        if pov.startswith("faction:"):
            try:
                return int(pov.split(":")[1])
            except ValueError:
                return None
        for e in self.frame.entities:
            if e.agent_id == pov:
                return e.faction
        return None

    def is_ally_of_pov(self, e: EntityView, view: ViewInfo) -> bool:
        if view.pov == "omniscient":
            return True
        pf = self._pov_faction(view)
        if pf is None:
            return True
        if e.faction == pf:
            return True
        rel = self.frame.relations if self.frame else []
        try:
            return rel[pf][e.faction] == 3
        except Exception:  # noqa: BLE001
            return False

    def hp_display(self, e: EntityView, view: ViewInfo) -> tuple[float, str | None]:
        """(fraction, label) - label None means 'show numbers'."""
        mx = max(1, e.max_hp)
        frac = max(0.0, min(1.0, e.hp / mx))
        if e.dead or not e.alive:
            return 0.0, "Dead" if self.is_ally_of_pov(e, view) else "Dead"
        if self.is_ally_of_pov(e, view):
            return frac, None
        if e.downed or e.hp <= 0:
            return 0.0, "Down"
        label = HP_BUCKETS[-1][1]
        for th, name in HP_BUCKETS:
            if frac >= th - 1e-9 and th > 0 or (th == 1.0 and frac >= 1.0):
                label = name
                break
        if frac >= 1.0:
            label = "Unhurt"
        elif frac > 0.75:
            label = "Scratched"
        elif frac > 0.5:
            label = "Wounded"
        elif frac > 0.25:
            label = "Bloodied"
        else:
            label = "Near death"
        q = 1.0 if frac >= 1 else 0.875 if frac > 0.75 else 0.625 if frac > 0.5 else 0.375 if frac > 0.25 else 0.125
        return q, label

    def entity_tip(self, e: EntityView, view: ViewInfo) -> Tip:
        fcol = self.faction_color(e.faction)
        if e.kind != "creature":
            lines = []
            if e.state:
                lines.append((f"State: {humanize(e.state)}", T.TEXT))
            if e.value_gp:
                lines.append((f"Worth about {e.value_gp} gp", T.YELLOW))
            return Tip(e.name, humanize(e.kind), lines, e.icon or "perspective-dice-six-faces-random", T.GOLD_HI, fcol if e.faction >= 0 else None)
        known = self.is_ally_of_pov(e, view)
        sub = " ".join(x for x in (e.race_name, e.class_name or humanize(e.template)) if x)
        if e.level and e.class_name and str(e.level) not in e.class_name:
            sub += f" {e.level}"
        if not e.level and e.cr:
            sub += f" · CR {e.cr:g}"
        frac, label = self.hp_display(e, view)
        if label is None:
            hp = (frac, f"{e.hp} / {e.max_hp}" + (f"  +{e.temp_hp}" if e.temp_hp else ""))
        else:
            hp = (frac, label)
        lines: list[tuple[str, T.Color]] = []
        lines.append((f"AC {e.ac if known else '?'}   ·   Speed {e.speed} ft   ·   {self.faction_name(e.faction)}", T.TEXT))
        if e.agent_id:
            lines.append((f"RL agent · {e.agent_id}", T.TEXT_FAINT))
        for c in e.conditions:
            lines.append((humanize(c), T.ORANGE))
            lines.append((cond_tip(c).lines[0][0], T.TEXT_DIM))
        for c in e.statuses:
            if c in CONDITIONS:
                lines.append((humanize(c), T.TEXT))
                lines.append((cond_tip(c).lines[0][0], T.TEXT_DIM))
        for f in e.effects:
            lines.append((humanize(f), T.CYAN))
            lines.append((cond_tip(f).lines[0][0], T.TEXT_DIM))
        if e.concentration:
            lines.append((f"Concentrating on {humanize(e.concentration)}", T.PURPLE))
        return Tip(e.name, sub, lines, e.icon, T.mix(fcol, (255, 255, 255), 0.35), fcol, hp=hp, width=300)

    def research_annotation(self):
        fr = self.frame
        if fr is None or not fr.annotations:
            return None
        cand = []
        act = self.ent(fr.active_id)
        if act is not None and act.agent_id:
            cand.append(act.agent_id)
        if self.view is not None:
            cand.append(self.view.pov)
        cand += list(fr.annotations)
        for c in cand:
            if c in fr.annotations:
                return c, fr.annotations[c]
        return None

    def outcome_kind(self, d: dict, frame) -> str:
        data = (d or {}).get("data") or {}
        winners = data.get("winners")
        player = {f.index for f in self.scene.factions if f.is_player_side} if self.scene else set()
        if winners:
            return "victory" if any(int(w) in player for w in winners) else "defeat"
        txt = (frame.outcome if frame else "") or str(data.get("reason", ""))
        low = txt.lower()
        if any(w in low for w in ("victory", "won", "success", "complete")):
            return "victory"
        if any(w in low for w in ("defeat", "lost", "wiped", "fail", "slain")):
            return "defeat"
        return "over"

    def _social(self, d: dict) -> None:
        t = d.get("t")
        src, dst = d.get("src", -1), d.get("dst", -1)
        amt = d.get("amount", 0) or 0
        key = (src, dst)
        if t == ET.DAMAGE and src >= 0 and dst >= 0:
            self.social.setdefault(key, {})["dmg"] = self.social.get(key, {}).get("dmg", 0) + amt
        elif t == ET.HEAL and src >= 0 and dst >= 0 and src != dst:
            self.social.setdefault(key, {})["heal"] = self.social.get(key, {}).get("heal", 0) + amt
        elif t == ET.GIVE and src >= 0 and dst >= 0:
            self.social.setdefault(key, {})["gold"] = self.social.get(key, {}).get("gold", 0) + amt
        elif t == ET.BETRAYAL and src >= 0 and dst >= 0:
            self.social.setdefault(key, {})["betray"] = self.social.get(key, {}).get("betray", 0) + 1

    def social_summary(self, eid: int) -> list[tuple[str, T.Color, str]]:
        out: list[tuple[str, T.Color, str]] = []

        def nm(i: int) -> str:
            return self._names.get(i, (f"#{i}", -1))[0]

        def top(field: str, as_src: bool):
            rows = []
            for (s, d), v in self.social.items():
                if field in v and ((s == eid) if as_src else (d == eid)) and s != d:
                    rows.append((v[field], d if as_src else s))
            rows.sort(reverse=True)
            return rows[:2]

        for amt, o in top("dmg", True):
            out.append(("sword-wound", T.ORANGE, f"Dealt {amt:g} damage to {nm(o)}"))
        for amt, o in top("dmg", False):
            out.append(("shield-impact", T.RED, f"Took {amt:g} damage from {nm(o)}"))
        for amt, o in top("heal", True):
            out.append(("heart-plus", T.GREEN, f"Healed {nm(o)} for {amt:g}"))
        for amt, o in top("heal", False):
            out.append(("heart-plus", T.GREEN, f"Healed by {nm(o)} for {amt:g}"))
        for amt, o in top("gold", True):
            out.append(("coins", T.YELLOW, f"Gave {amt:g} gp to {nm(o)}"))
        for amt, o in top("gold", False):
            out.append(("coins", T.YELLOW, f"Received {amt:g} gp from {nm(o)}"))
        for amt, o in top("betray", True):
            out.append(("dagger-rose", T.RED, f"Betrayed {nm(o)}"))
        for amt, o in top("betray", False):
            out.append(("dagger-rose", T.RED, f"Betrayed by {nm(o)}"))
        return out[:8]
