"""Human play: action bar, target mode, floor/token clicks, reaction modal."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from emberdeep.render.api import ActionDescriptor
from emberdeep.render.hud_api import UICommand
from emberdeep.render.ui import theme as T
from emberdeep.render.ui.combat_log import humanize
from emberdeep.render.ui.widgets import Tip, button

DIR_VEC = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
HOTKEY_LABELS = list("1234567890") + list("QWERTYUIOP")
HOTKEY_CODES = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                pygame.K_9, pygame.K_0, pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y,
                pygame.K_u, pygame.K_i, pygame.K_o, pygame.K_p]
KP_DIRS = {pygame.K_KP8: 0, pygame.K_KP9: 1, pygame.K_KP6: 2, pygame.K_KP3: 3, pygame.K_KP2: 4, pygame.K_KP1: 5,
           pygame.K_KP4: 6, pygame.K_KP7: 7}
COST_ORDER = ["action", "attack", "bonus", "reaction", "free", "movement", ""]
COST_LABEL = {"action": "ACTION", "attack": "ACTION", "bonus": "BONUS", "reaction": "REACTION", "free": "FREE",
              "movement": "MOVE", "": "OTHER"}
DECISION_TITLES = {1: "Opportunity Attack", 2: "Readied Action", 3: "Under Attack", 4: "Struck!", 5: "Spell Cast",
                   6: "Strike True", 7: "Protect an Ally", 8: "Truce Offer"}


def enabled(d: ActionDescriptor) -> bool:
    return bool(getattr(d, "enabled", True))


@dataclass
class Group:
    key: str
    label: str
    icon: str
    cost: str
    descs: list[ActionDescriptor] = field(default_factory=list)
    hotkey: str = ""
    code: int = -1

    @property
    def ok(self) -> bool:
        return any(enabled(d) for d in self.descs)


class ActionBar:
    def __init__(self, hud) -> None:
        self.hud = hud
        self.agent_id = ""
        self.actions: list[ActionDescriptor] = []
        self.groups: list[Group] = []
        self.moves: list[ActionDescriptor] = []
        self.approach: list[ActionDescriptor] = []
        self.says: list[ActionDescriptor] = []
        self.decisions: list[ActionDescriptor] = []
        self.end: ActionDescriptor | None = None
        self.sel: Group | None = None
        self.say_open = False
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.prompt_alpha = 0.0
        self.rects: dict[str, pygame.Rect] = {}      # last drawn button rects ('end', 'g:<key>', ...)
        self.age = 0.0

    @property
    def active(self) -> bool:
        return bool(self.actions)

    # ----------------------------------------------------------------- state
    def begin(self, agent_id: str, actions: list[ActionDescriptor]) -> None:
        self.agent_id = agent_id
        self.actions = list(actions)
        self.sel = None
        self.say_open = False
        self.age = 0.0
        self.groups, self.moves, self.approach, self.says, self.decisions = [], [], [], [], []
        self.end = None
        by: dict[str, Group] = {}
        order: list[Group] = []
        for d in self.actions:
            if d.kind == "end_turn":
                self.end = d
            elif d.kind == "move":
                self.moves.append(d)
            elif d.kind == "approach":
                self.approach.append(d)
            elif d.kind == "say":
                self.says.append(d)
            elif d.kind == "decision":
                self.decisions.append(d)
            else:
                key = d.ability_id or d.label.split(" ")[0] + ":" + d.cost
                g = by.get(key)
                if g is None:
                    g = Group(key, d.label, d.icon, d.cost)
                    by[key] = g
                    order.append(g)
                g.descs.append(d)
        order.sort(key=lambda g: COST_ORDER.index(g.cost) if g.cost in COST_ORDER else 99)
        for i, g in enumerate(order):
            if i < len(HOTKEY_LABELS):
                g.hotkey, g.code = HOTKEY_LABELS[i], HOTKEY_CODES[i]
        self.groups = order

    def end_choice(self) -> None:
        self.actions, self.groups, self.moves, self.approach, self.says, self.decisions = [], [], [], [], [], []
        self.end = None
        self.sel = None
        self.say_open = False

    def actor(self):
        fr = self.hud.frame
        if fr is None:
            return None
        for e in fr.entities:
            if e.agent_id == self.agent_id and self.agent_id:
                return e
        return self.hud.ent(fr.active_id)

    # ----------------------------------------------------------------- choosing
    def _choose(self, d: ActionDescriptor) -> UICommand | None:
        if not enabled(d):
            return None
        self.sel = None
        self.say_open = False
        a = self.hud.audio
        if a is not None:
            try:
                a.on_ui("click")
            except Exception:  # noqa: BLE001
                pass
        return UICommand("choose_action", int(d.index))

    def _targets(self, g: Group) -> list[ActionDescriptor]:
        actor = self.actor()
        aid = actor.id if actor else -2
        return [d for d in g.descs if enabled(d) and ((d.target_eid not in (-1, aid)) or d.target_cell is not None)]

    def activate(self, g: Group) -> UICommand | None:
        if not g.ok:
            return None
        tg = self._targets(g)
        if not tg:
            ds = [d for d in g.descs if enabled(d)]
            return self._choose(ds[0]) if ds else None
        self.sel = g if self.sel is not g else None
        self.say_open = False
        return None

    def cancel(self) -> bool:
        if self.sel is not None:
            self.sel = None
            return True
        if self.say_open:
            self.say_open = False
            return True
        return False

    # ----------------------------------------------------------------- input
    def key(self, ev) -> UICommand | None:
        k = ev.key
        if self.decisions:
            for i, d in enumerate(self.decisions):
                if k in (HOTKEY_CODES[i] if i < len(HOTKEY_CODES) else -1,):
                    return self._choose(d)
            return None
        if k == pygame.K_ESCAPE:
            return UICommand("noop") if self.cancel() else None
        for g in self.groups:
            if g.code == k:
                return self.activate(g) or UICommand("noop")
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.end is not None:
            return self._choose(self.end)
        if k in KP_DIRS:
            d = KP_DIRS[k]
            for m in self.moves:
                if m.direction == d:
                    return self._choose(m)
        return None

    def world_click(self, cell, eid: int, button_: int, view) -> UICommand | None:
        if not self.active or self.decisions:
            return None
        if button_ == 3:
            if self.cancel():
                return UICommand("noop")
            return None
        if button_ != 1:
            return None
        if self.sel is not None:
            d = self._match_target(self.sel, cell, eid)
            if d is not None:
                return self._choose(d)
            # approach for the selected ability?
            if eid >= 0:
                ap = [a for a in self.approach if a.target_eid == eid and (not a.ability_id or a.ability_id == self.sel.key)]
                if ap:
                    return self._choose(ap[0])
            return UICommand("noop")
        # no ability selected
        if eid >= 0:
            ds = [d for g in self.groups for d in g.descs if d.target_eid == eid and enabled(d)]
            actor = self.actor()
            if actor is not None and eid == actor.id:
                return None
            if ds:
                return self._choose(ds[0])
            ap = [a for a in self.approach if a.target_eid == eid]
            if ap:
                return self._choose(ap[0])
            return None
        mv = self._nearest_move(cell)
        if mv is not None:
            return self._choose(mv)
        return None

    def _match_target(self, g: Group, cell, eid: int) -> ActionDescriptor | None:
        ds = self._targets(g)
        if eid >= 0:
            for d in ds:
                if d.target_eid == eid:
                    return d
            e = self.hud.ent(eid)
            if e is not None:
                cell = (e.x, e.y)
        if cell is not None:
            best, bd = None, 1e9
            for d in ds:
                if d.target_cell is None:
                    continue
                dd = max(abs(d.target_cell[0] - cell[0]), abs(d.target_cell[1] - cell[1]))
                if dd < bd:
                    best, bd = d, dd
            if best is not None and (bd <= 0 or any(c == tuple(cell) for c in best.area_cells) or bd <= 1):
                return best
        return None

    def _nearest_move(self, cell) -> ActionDescriptor | None:
        if cell is None:
            return None
        cands = [m for m in (self.moves + self.approach) if enabled(m)]
        best, bd = None, 1e9
        for m in cands:
            if m.target_cell is not None:
                dd = math.hypot(m.target_cell[0] - cell[0], m.target_cell[1] - cell[1])
                if dd < bd:
                    best, bd = m, dd
        if best is not None:
            return best
        actor = self.actor()
        if actor is None:
            return None
        vx, vy = cell[0] - actor.x, cell[1] - actor.y
        if vx == 0 and vy == 0:
            return None
        ang = math.atan2(vy, vx)
        bestd = 9.0
        for m in self.moves:
            if 0 <= m.direction < 8:
                dx, dy = DIR_VEC[m.direction]
                da = abs((ang - math.atan2(dy, dx) + math.pi) % (2 * math.pi) - math.pi)
                if da < bestd:
                    best, bestd = m, da
        return best

    # ----------------------------------------------------------------- tooltips
    def _tip(self, g: Group) -> Tip:
        lines: list[tuple[str, T.Color]] = []
        d0 = g.descs[0]
        if d0.tooltip:
            lines.append((d0.tooltip, T.TEXT_DIM))
        hc = [d.hit_chance for d in g.descs if d.hit_chance is not None]
        ed = [d.expected_damage for d in g.descs if d.expected_damage is not None]
        if hc:
            lines.append((f"Hit chance: {max(hc) * 100:.0f}%" + (f" (best of {len(hc)})" if len(hc) > 1 else ""), T.GREEN))
        if ed:
            lines.append((f"Expected damage: {max(ed):.1f}", T.ORANGE))
        n = len(self._targets(g))
        if n:
            lines.append((f"{n} valid target{'s' if n != 1 else ''}", T.TEXT_DIM))
        if not g.ok:
            lines.append(("Currently unavailable.", T.RED))
        lines.append((f"Hotkey: {g.hotkey}" if g.hotkey else "", T.TEXT_FAINT))
        return Tip(g.label, COST_LABEL.get(g.cost, "").title(), [ln for ln in lines if ln[0]], g.icon or "sparkles",
                   T.COST_COLORS.get(g.cost, T.GOLD_HI), T.COST_COLORS.get(g.cost))

    # ----------------------------------------------------------------- draw
    def draw(self, surf: pygame.Surface, area: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        if not self.active:
            return
        self.age += hud.dt
        self.rects = {}
        if self.decisions:
            self._modal(surf, view)
            return
        pad = ui.s(10)
        bs = ui.s(52)
        gapb = ui.s(6)
        grp_gap = ui.s(16)
        move_w = ui.s(126)
        end_w = ui.s(124)
        inner_w = area.w - 2 * pad - move_w - end_w - ui.s(24)
        # arrange groups into rows
        rows: list[list[Group]] = [[]]
        x = 0
        last_cost = None
        for g in self.groups:
            extra = grp_gap if (last_cost is not None and g.cost != last_cost) else gapb
            if rows[-1] and x + extra + bs > inner_w:
                rows.append([])
                x = 0
                extra = 0
            if not rows[-1]:
                extra = 0
            x += extra + bs
            rows[-1].append(g)
            last_cost = g.cost
        label_h = ui.s(14)
        row_h = bs + label_h + ui.s(6)
        H = pad * 2 + row_h * len(rows) - ui.s(6) + ui.s(2)
        H = max(H, ui.s(96))
        used_w = max((self._row_w(r, bs, gapb, grp_gap) for r in rows), default=0)
        W = min(area.w, used_w + move_w + end_w + 2 * pad + ui.s(24))
        panel = pygame.Rect(0, 0, W, H)
        panel.midbottom = (area.centerx, area.bottom)
        self.rect = panel
        ui.draw_panel(surf, panel, "panel", accent=T.GOLD)
        hud.hot(panel, block=True)
        actor = self.actor()
        # movement indicator (left)
        mr = pygame.Rect(panel.x + pad, panel.y + pad, move_w, panel.h - 2 * pad)
        self._movement(surf, mr, actor)
        sepx = mr.right + ui.s(6)
        pygame.draw.line(surf, T.GOLD_DIM, (sepx, panel.y + pad), (sepx, panel.bottom - pad), 1)
        # ability rows
        y = panel.y + pad
        for row in rows:
            x = sepx + ui.s(12)
            prev = None
            gx0 = x
            for g in row:
                if prev is not None:
                    x += grp_gap if g.cost != prev else gapb
                    if g.cost != prev:
                        self._label(surf, COST_LABEL.get(prev, ""), gx0, x - grp_gap // 2, y, prev)
                        gx0 = x
                elif g.cost is not None:
                    gx0 = x
                self._ability_button(surf, pygame.Rect(x, y + label_h, bs, bs), g, view)
                x += bs
                prev = g.cost
            if prev is not None:
                self._label(surf, COST_LABEL.get(prev, ""), gx0, x, y, prev)
            y += row_h
        # right side: End Turn + Say
        er = pygame.Rect(panel.right - pad - end_w, panel.y + pad, end_w, ui.s(40))
        endok = self.end is not None
        self.rects['end'] = er
        button(hud, surf, er, label="End Turn", icon="hourglass", pt=13.5, enabled=endok, active=False,
               click=(lambda: self._choose(self.end)) if endok else None, color=T.GOLD_HI,
               tip=Tip("End Turn", lines=[("Finish this creature's turn.", T.TEXT_DIM), ("Hotkey: Enter", T.TEXT_FAINT)]))
        if self.says:
            sr = pygame.Rect(er.x, er.bottom + ui.s(8), er.w, ui.s(28))
            button(hud, surf, sr, label="Speak…", icon="chat-bubble", pt=12.5, active=self.say_open,
                   click=self._toggle_say, tip=Tip("Say something", lines=[("Free action; talk to allies and foes.", T.TEXT_DIM)]))
            if self.say_open:
                self._say_menu(surf, sr)
        # target mode prompt + highlights
        if self.sel is not None:
            self._target_mode(surf, panel, view)
        elif view.hovered_cell is not None or view.hovered_eid >= 0:
            self._hover_hint(surf, view)

    def _row_w(self, row, bs, gapb, grp_gap) -> int:
        x = 0
        prev = None
        for g in row:
            if prev is not None:
                x += grp_gap if g.cost != prev else gapb
            x += bs
            prev = g.cost
        return x

    def _label(self, surf, text: str, x0: int, x1: int, y: int, cost: str) -> None:
        ui = self.hud.ui
        col = T.COST_COLORS.get(cost, T.TEXT_DIM)
        t = ui.text(text, "title", 9, col, spacing=1)
        cx = (x0 + x1) // 2
        surf.blit(t, t.get_rect(midtop=(cx, y)))
        ly = y + ui.s(5)
        if x1 - x0 > t.get_width() + ui.s(16):
            pygame.draw.line(surf, T.scale_color(col, 0.5), (x0, ly), (cx - t.get_width() // 2 - ui.s(5), ly), 1)
            pygame.draw.line(surf, T.scale_color(col, 0.5), (cx + t.get_width() // 2 + ui.s(5), ly), (x1, ly), 1)

    def _ability_button(self, surf, r: pygame.Rect, g: Group, view) -> None:
        hud, ui = self.hud, self.hud.ui
        ok = g.ok
        sel = self.sel is g
        self.rects['g:' + g.key] = r
        cc = T.COST_COLORS.get(g.cost, T.GREY)
        hov = ok and r.collidepoint(hud.mouse) and not hud.mouse_blocked(r)
        pygame.draw.rect(surf, (30, 33, 46) if not (hov or sel) else (46, 50, 68), r, border_radius=ui.s(7))
        if sel:
            gl = pygame.Surface((r.w + ui.s(12), r.h + ui.s(12)), pygame.SRCALPHA)
            pygame.draw.rect(gl, (*T.GOLD_HI, 90), gl.get_rect(), border_radius=ui.s(10), width=ui.s(3))
            surf.blit(gl, (r.x - ui.s(6), r.y - ui.s(6)))
        bc = T.GOLD_HI if (sel or hov) else (T.scale_color(cc, 0.75) if ok else (74, 70, 64))
        pygame.draw.rect(surf, bc, r, width=max(1, ui.s(2)) if sel else max(1, ui.s(1)), border_radius=ui.s(7))
        ic = ui.icon(g.icon or ("sparkles" if g.cost in ("action", "attack") else "perspective-dice-six-faces-random"),
                     int(r.w * 0.62), T.TEXT if ok else (92, 90, 90))
        surf.blit(ic, ic.get_rect(center=(r.centerx, r.centery - ui.s(1))))
        # cost pip
        pr = pygame.Rect(0, 0, ui.s(9), ui.s(9))
        pr.bottomright = (r.right - ui.s(4), r.bottom - ui.s(4))
        col = cc if ok else (80, 76, 70)
        if g.cost == "bonus":
            pygame.draw.polygon(surf, col, [(pr.centerx, pr.y), (pr.right, pr.bottom), (pr.x, pr.bottom)])
        elif g.cost == "reaction":
            pygame.draw.circle(surf, col, pr.center, pr.w // 2, max(1, ui.s(2)))
        elif g.cost == "free":
            pygame.draw.rect(surf, col, pr.inflate(-2, -2), width=1)
        else:
            pygame.draw.circle(surf, col, pr.center, pr.w // 2)
        if g.hotkey:
            ui.blit_text(surf, g.hotkey, (r.x + ui.s(5), r.y + ui.s(3)), "mono", 9.5, T.GOLD if ok else T.TEXT_FAINT)
        n = len(self._targets(g))
        if n > 1:
            ui.blit_text(surf, f"×{n}", (r.x + ui.s(5), r.bottom - ui.s(4)), "mono", 9, T.TEXT_DIM, anchor="bottomleft")
        if not ok:
            ov = pygame.Surface(r.size, pygame.SRCALPHA)
            pygame.draw.rect(ov, (10, 10, 14, 110), ov.get_rect(), border_radius=ui.s(7))
            surf.blit(ov, r.topleft)
        hud.hot(r, tip=self._tip(g), click=(lambda g=g: self.activate(g)) if ok else None)

    def _movement(self, surf, r: pygame.Rect, actor) -> None:
        hud, ui = self.hud, self.hud.ui
        ui.blit_text(surf, "MOVEMENT", (r.centerx, r.y + ui.s(6)), "title", 9.5, T.COST_COLORS["movement"], anchor="center", spacing=1)
        cy = r.y + ui.s(34)
        ui.blit_icon(surf, "boot-prints", (r.x + ui.s(16), cy), ui.s(28), T.COST_COLORS["movement"])
        left = actor.movement_left if actor else 0
        spd = actor.speed if actor else 0
        ui.blit_text(surf, f"{left}", (r.x + ui.s(38), cy - ui.s(2)), "mono", 17, T.TEXT, anchor="midleft")
        ui.blit_text(surf, f"/ {spd} ft", (r.x + ui.s(38) + ui.text_w(f"{left}", "mono", 17) + ui.s(4), cy + ui.s(2)),
                     "mono", 10, T.TEXT_DIM, anchor="midleft")
        bar = pygame.Rect(r.x + ui.s(2), cy + ui.s(22), r.w - ui.s(4), ui.s(8))
        ui.bar(surf, bar, left / max(1, spd), T.COST_COLORS["movement"])
        ui.blit_text(surf, "click a tile · numpad" if self.moves else "no moves", (r.centerx, bar.bottom + ui.s(10)), "body_italic", 10.5,
                     T.TEXT_FAINT, anchor="center")

    def _toggle_say(self):
        self.say_open = not self.say_open

    def _say_menu(self, surf, anchor: pygame.Rect) -> None:
        hud, ui = self.hud, self.hud.ui
        ih = ui.s(26)
        items = self.says[:12]
        w = ui.s(230)
        h = ih * len(items) + ui.s(12)
        r = pygame.Rect(0, 0, w, h)
        r.bottomright = (anchor.right, self.rect.y - ui.s(8))
        ui.draw_panel(surf, r, "modal")
        hud.hot(r, block=True)
        for i, d in enumerate(items):
            ir = pygame.Rect(r.x + ui.s(6), r.y + ui.s(6) + i * ih, r.w - ui.s(12), ih - ui.s(2))
            lab = d.label[4:] if d.label.lower().startswith("say:") else d.label
            self.rects[f's:{i}'] = ir
            button(hud, surf, ir, label=lab.strip(), pt=12.5, role="body_italic", click=lambda d=d: self._choose(d))

    def _hover_hint(self, surf, view) -> None:
        """When no ability is selected, preview where a click would walk."""
        hud, ui = self.hud, self.hud.ui
        cell = view.hovered_cell
        if cell is None or view.hovered_eid >= 0:
            return
        if hud.mouse_over_panel():
            return
        mv = self._nearest_move(cell)
        if mv is None or mv.target_cell is None:
            return
        self._cell_box(surf, view, mv.target_cell, T.COST_COLORS["movement"], 70)
        actor = self.actor()
        if actor is not None:
            a = view.world_to_screen(actor.x + actor.size / 2, actor.y + actor.size / 2)
            b = view.world_to_screen(mv.target_cell[0] + 0.5, mv.target_cell[1] + 0.5)
            pygame.draw.line(surf, (*T.COST_COLORS["movement"],), a, b, max(2, ui.s(2)))

    def _cell_box(self, surf, view, cell, col, alpha, width: int = 2) -> None:
        x, y = view.world_to_screen(cell[0], cell[1])
        x2, y2 = view.world_to_screen(cell[0] + 1, cell[1] + 1)
        r = pygame.Rect(int(x), int(y), max(2, int(x2 - x)), max(2, int(y2 - y)))
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        s.fill((*col, alpha))
        surf.blit(s, r.topleft)
        pygame.draw.rect(surf, col, r, width=width)

    def _area_preview(self, surf, view, cells, col) -> None:
        """Translucent fill for an area of effect with a single outline around its boundary."""
        if not cells:
            return
        cs = {(int(c[0]), int(c[1])) for c in cells}
        px = lambda x, y: view.world_to_screen(x, y)  # noqa: E731
        for (cx, cy) in cs:
            x0, y0 = px(cx, cy)
            x1, y1 = px(cx + 1, cy + 1)
            r = pygame.Rect(int(x0), int(y0), max(2, int(x1 - x0) + 1), max(2, int(y1 - y0) + 1))
            s = pygame.Surface(r.size, pygame.SRCALPHA)
            s.fill((*col, 64))
            surf.blit(s, r.topleft)
        w = max(2, self.hud.ui.s(2))
        for (cx, cy) in cs:
            x0, y0 = px(cx, cy)
            x1, y1 = px(cx + 1, cy + 1)
            if (cx, cy - 1) not in cs:
                pygame.draw.line(surf, col, (x0, y0), (x1, y0), w)
            if (cx, cy + 1) not in cs:
                pygame.draw.line(surf, col, (x0, y1), (x1, y1), w)
            if (cx - 1, cy) not in cs:
                pygame.draw.line(surf, col, (x0, y0), (x0, y1), w)
            if (cx + 1, cy) not in cs:
                pygame.draw.line(surf, col, (x1, y0), (x1, y1), w)

    def _target_mode(self, surf, panel: pygame.Rect, view) -> None:
        hud, ui = self.hud, self.hud.ui
        g = self.sel
        ds = self._targets(g)
        pulse = 0.5 + 0.5 * math.sin(hud.t * 5)
        col = T.ORANGE if g.cost in ("action", "attack") else T.COST_COLORS.get(g.cost, T.GOLD_HI)
        # prompt
        txt = f"Choose a target for {g.label}"
        t = ui.text(txt, "title", 14, T.TEXT, shadow=True)
        sub = ui.text("click a highlighted target · right-click or Esc to cancel", "body_italic", 12, T.TEXT_DIM)
        pw = max(t.get_width(), sub.get_width()) + ui.s(36)
        pr = pygame.Rect(0, 0, pw, ui.s(48))
        pr.midbottom = (panel.centerx, panel.y - ui.s(10))
        ui.draw_panel(surf, pr, "plate", accent=col)
        surf.blit(t, t.get_rect(midtop=(pr.centerx, pr.y + ui.s(7))))
        surf.blit(sub, sub.get_rect(midbottom=(pr.centerx, pr.bottom - ui.s(6))))
        hov_e, hov_c = view.hovered_eid, view.hovered_cell
        hover_d = None
        if not hud.mouse_over_panel():
            hover_d = self._match_target(g, hov_c, hov_e)
        for d in ds:
            if d.target_eid >= 0 and d.target_eid in view.entity_screen:
                cx, cy, r = view.entity_screen[d.target_eid]
                is_h = d is hover_d
                rr = r + ui.s(5) + (ui.s(3) * pulse if is_h else 0)
                gl = pygame.Surface((int(rr * 2 + ui.s(16)), int(rr * 2 + ui.s(16))), pygame.SRCALPHA)
                c = gl.get_rect().center
                pygame.draw.circle(gl, (*col, 60 if not is_h else 120), c, int(rr + ui.s(4)))
                pygame.draw.circle(gl, (*col, 255), c, int(rr), max(2, ui.s(3 if is_h else 2)))
                surf.blit(gl, gl.get_rect(center=(int(cx), int(cy))))
                if d.hit_chance is not None:
                    lab = ui.text(f"{d.hit_chance * 100:.0f}%", "mono", 12, T.TEXT, shadow=True)
                    br = lab.get_rect(midtop=(int(cx), int(cy + r + ui.s(6))))
                    pygame.draw.rect(surf, (14, 15, 22), br.inflate(ui.s(10), ui.s(4)), border_radius=ui.s(8))
                    pygame.draw.rect(surf, col, br.inflate(ui.s(10), ui.s(4)), width=1, border_radius=ui.s(8))
                    surf.blit(lab, br)
            elif d.target_cell is not None:
                self._cell_box(surf, view, d.target_cell, col, 40 + int(30 * pulse), 2)
        if hover_d is not None:
            self._area_preview(surf, view, hover_d.area_cells, T.RED if g.cost != "bonus" else col)
            if hover_d.target_cell is not None:
                self._cell_box(surf, view, hover_d.target_cell, T.GOLD_HI, 30, 3)

    # ----------------------------------------------------------------- reaction modal
    def _modal(self, surf, view) -> None:
        hud, ui = self.hud, self.hud.ui
        W, H = surf.get_size()
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((4, 4, 8, int(150 * min(1.0, self.age / 0.25))))
        surf.blit(veil, (0, 0))
        dec = hud.frame.decision if hud.frame else None
        kind = dec.get("kind") if dec else None
        title = DECISION_TITLES.get(kind, humanize(str(kind))) if isinstance(kind, int) or kind is None else humanize(str(kind))
        w = ui.s(420)
        n = len(self.decisions)
        ih = ui.s(40)
        h = ui.s(92) + n * (ih + ui.s(8))
        r = pygame.Rect(0, 0, w, h)
        r.center = (W // 2, int(H * 0.42))
        k = T.ease_out_cubic(self.age / 0.3)
        r.y += int((1 - k) * ui.s(16))
        ui.draw_panel(surf, r, "modal", accent=T.PURPLE)
        hud.hot(pygame.Rect(0, 0, W, H), block=True)
        hud.hot(r, block=True)
        ui.blit_text(surf, "REACTION", (r.centerx, r.y + ui.s(22)), "title", 10.5, T.PURPLE, anchor="center", spacing=3)
        ui.blit_text(surf, title, (r.centerx, r.y + ui.s(44)), "title_bold", 21, T.GOLD_HI, anchor="center", shadow=True)
        ui.divider(surf, r.x + ui.s(30), r.right - ui.s(30), r.y + ui.s(64), T.GOLD, 120)
        opts = (dec or {}).get("options") or []
        y = r.y + ui.s(78)
        for i, d in enumerate(self.decisions):
            br = pygame.Rect(r.x + ui.s(28), y, r.w - ui.s(56), ih)
            self.rects[f'd:{i}'] = br
            label = d.label or (opts[i] if i < len(opts) else f"Option {i + 1}")
            button(hud, surf, br, label=label, icon=d.icon, pt=14, hotkey=HOTKEY_LABELS[i] if i < 20 else "",
                   click=lambda d=d: self._choose(d), tip=Tip(label, lines=[(d.tooltip, T.TEXT_DIM)]) if d.tooltip else None)
            y += ih + ui.s(8)
