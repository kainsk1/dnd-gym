"""World -> render contract adapter (``emberdeep.render.api``) and the ``render()`` bridge.

* :class:`FrameBuilder` converts an Emberdeep world into a ``StaticScene`` (once per episode) and a ``FrameState``
  (after every step) - the only objects the renderer, replays and the viewer ever see.
* :func:`describe_actions` explains every currently legal flat action as an ``ActionDescriptor`` (labels, icons,
  costs, targets, area previews, hit chance / expected damage) for the human action bar and for debugging.
* :class:`RenderBridge` owns the lazily created ``emberdeep.render.Renderer`` for ``render_mode`` ``human`` /
  ``rgb_array`` (pygame / moderngl are imported only here, never in training mode).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import (
    ActionCost, AreaShape, Condition, ControllerKind, DecisionKind, DoorState, EntityKind, GameMode, Relation,
    Status, DIR_VECTORS,
)
from emberdeep.env.rewards import COMPONENTS
from emberdeep.render.api import (
    ActionDescriptor, EntityView, FactionView, FrameState, LightSpec, StaticScene, ZoneView,
)

log = logging.getLogger(__name__)

_CONDITION_NAMES = [c.name.lower() for c in Condition]
_STATUS_BITS = [(int(s.value), s.name.lower()) for s in Status if s.value and (s.value & (s.value - 1)) == 0]
_AMBIENT = {"dark": (0.06, 0.06, 0.09), "dim": (0.15, 0.15, 0.20), "bright": (0.60, 0.62, 0.66)}
_THEME_BRIGHT = {"forest": (0.56, 0.63, 0.54), "arena": (0.66, 0.62, 0.55), "castle": (0.58, 0.58, 0.62)}
# light kind -> (flicker, height, intensity)
_LIGHT_STYLE = {"torch": (0.15, 1.2, 1.0), "brazier": (0.25, 1.1, 1.0), "candle": (0.20, 1.0, 0.9),
                "fire": (0.30, 1.0, 1.0), "lava": (0.35, 0.8, 0.85), "crystal": (0.05, 1.0, 1.0),
                "magic": (0.05, 1.0, 1.0), "sunbeam": (0.0, 2.0, 0.9)}
_PC_TINTS = {
    "barbarian": (150, 90, 60), "bard": (170, 110, 160), "cleric": (190, 170, 90), "druid": (90, 140, 80),
    "fighter": (130, 130, 150), "monk": (200, 150, 80), "paladin": (200, 200, 220), "ranger": (80, 120, 80),
    "rogue": (70, 90, 100), "sorcerer": (180, 70, 90), "warlock": (110, 70, 150), "wizard": (90, 90, 180),
}
_COST_NAME = {ActionCost.ACTION: "action", ActionCost.BONUS_ACTION: "bonus", ActionCost.REACTION: "reaction",
              ActionCost.FREE: "free", ActionCost.OBJECT_INTERACTION: "free", ActionCost.ATTACK: "attack",
              ActionCost.MOVEMENT: "movement", ActionCost.LEGENDARY: "legendary", ActionCost.SPECIAL: "special"}
_DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_DOOR_STATE = {int(DoorState.OPEN): "open", int(DoorState.CLOSED): "closed", int(DoorState.LOCKED): "locked",
               int(DoorState.BROKEN): "broken"}


def _status_names(mask: int) -> list[str]:
    return [n for bit, n in _STATUS_BITS if mask & bit]


def _cond_names(mask: int) -> list[str]:
    return [n for i, n in enumerate(_CONDITION_NAMES) if (mask >> i) & 1]


class FrameBuilder:
    """Builds ``StaticScene`` / ``FrameState`` objects for an :class:`EmberdeepCore`."""

    def __init__(self, core):
        self.core = core
        self._class_icons: dict[str, str] = {}
        self.faction_explored: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------------------------ scene
    def scene(self) -> StaticScene:
        core = self.core
        w = core.world
        info = core.info
        md = info.map_data
        spec = info.spec
        lights = []
        for ld in md.lights:
            fl, ht, it = _LIGHT_STYLE.get(ld.kind, _LIGHT_STYLE["torch"])
            bright, dim = ld.bright_ft / 5.0, ld.dim_ft / 5.0
            if ld.kind == "lava" and bright <= 0:
                bright, dim = 1.5, max(dim, 3.0)
            lights.append(LightSpec(float(ld.x), float(ld.y), bright, dim, tuple(ld.color), it, fl, ht, ld.kind))
        theme = md.theme
        amb = _AMBIENT.get(md.ambient, _AMBIENT["dark"])
        if md.ambient == "bright":
            amb = _THEME_BRIGHT.get(theme, amb)
        facs = []
        for i, f in enumerate(info.factions):
            facs.append(FactionView(i, f.id, f.name, tuple(int(v) for v in f.color),
                                    bool(f.player_side) or i in info.player_factions))
        return StaticScene(
            name=spec.name, width=int(md.width), height=int(md.height), terrain=np.array(md.terrain, np.uint8),
            material=np.array(md.material, np.uint8), elevation=None, decor=list(md.decor), lights=lights,
            ambient=tuple(amb), theme=theme, factions=facs, agent_ids=list(core.agents),
            description=spec.description, seed=int(w.seed), comm_vocab=list(info.comm_vocab),
            regions={k: [tuple(c) for c in v] for k, v in w.regions.items()})

    def reset(self) -> None:
        self.faction_explored = {}

    # ------------------------------------------------------------------------------------ frames
    def _class_icon(self, cid: str) -> str:
        ic = self._class_icons.get(cid)
        if ic is None:
            try:
                from emberdeep import data
                ic = data.get_class(cid).icon
            except Exception:  # noqa: BLE001
                ic = ""
            self._class_icons[cid] = ic
        return ic

    def creature_view(self, c, ret_raw=None) -> EntityView:
        core = self.core
        w = core.world
        from emberdeep.core import resolution as R
        agent = core.eid_agent.get(c.eid)
        ctrl = {ControllerKind.AGENT: "agent", ControllerKind.SCRIPTED: "scripted",
                ControllerKind.HUMAN: "human"}.get(c.controller, "scripted")
        md = c.monster_def
        if c.is_pc:
            primary = max(c.class_levels.items(), key=lambda kv: kv[1])[0] if c.class_levels else c.template_id
            icon = self._class_icon(primary) or "person"
            tint = _PC_TINTS.get(primary, (140, 140, 140))
            cls_name = " / ".join(f"{k.capitalize()} {v}" for k, v in c.class_levels.items()) or c.template_id
        else:
            icon = getattr(md, "icon", "") or "monster-grasp"
            tint = tuple(getattr(md, "token_tint", (128, 128, 128)))
            cls_name = getattr(md, "name", c.name)
        status = int(c.status)
        res: dict[str, tuple[int, int]] = {}
        for i in range(9):
            if c.spell_slots_max[i]:
                res[f"slot_{i + 1}"] = (int(c.spell_slots[i]), int(c.spell_slots_max[i]))
        if c.pact_max:
            res["pact"] = (int(c.pact_slots), int(c.pact_max))
        for k, mx in c.resource_max.items():
            res[k] = (int(c.resources.get(k, 0)), int(mx))
        pots = sum(i.qty for i in c.inventory if getattr(i.def_, "heal", None))
        if pots:
            res["potions"] = (int(pots), int(pots))
        stats: dict[str, float] = {k: float(v) for k, v in c.stats.items() if isinstance(v, (int, float))}
        for nm, v in zip(("str", "dex", "con", "int", "wis", "cha"), c.scores):
            stats[nm] = float(v)
        if agent is not None:
            ai = core.agent_index[agent]
            tot = core.rewards.totals[ai]
            for j, name in enumerate(COMPONENTS):
                if tot[j]:
                    stats["r_" + name] = float(tot[j])
            stats["return"] = float(core.rewards.ret[ai])
            stats["return_raw"] = float(core.rewards.ret_raw[ai])
            h = core.history
            if h.cap > c.eid:
                stats["dmg_dealt_total"] = float(h.dmg[c.eid].sum())
                stats["healing_given_total"] = float(h.heal[c.eid].sum())
                stats["betrayals_committed"] = float(h.betray[c.eid].sum())
        try:
            speed = R.base_speed(w, c)
        except Exception:  # noqa: BLE001
            speed = c.speeds.get("walk", 30)
        active = w.active == c.eid and not w.done
        dead = bool(status & int(S.DEAD))
        return EntityView(
            id=c.eid, kind="creature", name=c.name, x=int(c.x), y=int(c.y), z=int(c.z), size=int(c.size_sq),
            template=c.template_id, icon=icon, faction=int(c.faction), tint=tuple(int(v) for v in tint),
            agent_id=agent, controller=ctrl, hp=int(c.hp), max_hp=int(c.max_hp), temp_hp=int(c.temp_hp),
            ac=int(R.compute_ac(w, c)), level=int(c.level) if c.is_pc else 0, cr=float(c.cr),
            class_name=cls_name, race_name=c.race or "", conditions=_cond_names(int(c.cond_mask)),
            effects=[e.ID for e in c.effects], statuses=_status_names(status),
            concentration=(c.concentration.spell_id if c.concentration is not None else None),
            alive=not dead, downed=bool(status & int(S.DYING | S.STABLE)) and not dead, dead=dead,
            hidden=bool(status & int(S.HIDDEN)), death_saves=(int(c.death_succ), int(c.death_fail)),
            speed=int(speed), movement_left=int(c.ts.movement_left) if active else 0, resources=res,
            gold=int(c.gold), inventory=[(i.name if i.qty == 1 else f"{i.name} x{i.qty}") for i in c.inventory],
            light=(tuple(c.light) if c.light is not None else None), facing=0.0, stats=stats)

    def object_view(self, o) -> EntityView:
        k = int(o.kind)
        state, value, icon = "", 0, ""
        if k == int(EntityKind.DOOR):
            kind, state = "door", _DOOR_STATE.get(int(o.state), "closed")
        elif k == int(EntityKind.CHEST):
            kind = "chest"
            state = "open" if o.open else ("locked" if o.locked else "closed")
            value = int(o.gold + sum(i.value_gp for i in o.items))
        elif k == int(EntityKind.LOOT):
            kind, value = "loot", int(o.value_gp)
        elif k == int(EntityKind.LEVER):
            kind, state = "lever", ("on" if o.on else "off")
        elif k == int(EntityKind.TRAP):
            kind = "trap"
            state = "disarmed" if o.disarmed else ("triggered" if o.triggered else "armed")
        elif k == int(EntityKind.ALTAR):
            kind, state = "altar", f"{o.charges}/{o.max_charges}" if o.max_charges else ""
        else:
            kind = "exit"
        return EntityView(id=o.eid, kind=kind, name=o.name, x=int(o.x), y=int(o.y), icon=icon, state=state,
                          value_gp=value)

    def frame(self, events=(), out=None, annotations: dict | None = None, step: int | None = None) -> FrameState:
        core = self.core
        w = core.world
        H, W = w.map.H, w.map.W
        ents: list[EntityView] = []
        for c in w.creatures:
            ents.append(self.creature_view(c))
        for o in w.objects:
            if not o.present:
                continue
            if o.kind == int(EntityKind.TRAP) and o.hidden:
                continue
            ents.append(self.object_view(o))
        zones = [ZoneView(int(z.zid), z.kind, [tuple(c) for c in z.cells], tuple(z.center), float(z.radius),
                          int(z.source), bool(z.harmful), int(z.attached_to)) for z in w.zones]
        full = np.ones((H, W), bool)
        visible: dict[str, np.ndarray] = {"omniscient": full}
        explored: dict[str, np.ndarray] = {"omniscient": full}
        for a in core.agents:
            if core.is_live(a):
                core.update_explored(a)
                visible[a] = np.array(core.visible_for(a), bool)
            else:
                visible[a] = np.zeros((H, W), bool)
            explored[a] = core.explored[a].copy()
        for f in range(len(w.factions)):
            tv = w.team_visible(f)
            fe = self.faction_explored.get(f)
            if fe is None:
                fe = np.zeros((H, W), bool)
                self.faction_explored[f] = fe
            fe |= tv
            visible[f"faction:{f}"] = tv
            explored[f"faction:{f}"] = fe.copy()
        doors = {(int(d.x), int(d.y)): _DOOR_STATE.get(int(d.state), "closed") for d in w.objects
                 if d.kind == int(EntityKind.DOOR) and d.present}
        vocab = core.info.comm_vocab
        msgs = []
        for e in events:
            if e.t == ET.SAY:
                tok = int(e.kind)
                msgs.append((int(e.src), tok, vocab[tok] if 0 <= tok < len(vocab) else f"#{tok}"))
        rewards, returns = {}, {}
        for a in core.agents:
            i = core.agent_index[a]
            returns[a] = float(core.rewards.ret[i])
            rewards[a] = float(out.rewards[i]) if out is not None else 0.0
        objectives = []
        for st in w.objectives:
            status = "done" if st.done else ("failed" if st.failed else "active")
            objectives.append((st.spec.name, status))
        dec = w.pending_decision
        decision = None
        if dec is not None:
            decision = {"eid": int(dec.eid), "kind": DecisionKind(int(dec.kind)).name.lower(),
                        "options": [o.label or o.key for o in dec.options]}
        outcome_text = ""
        metrics: dict[str, float] = {}
        if w.done:
            oc = w.outcome
            win = ", ".join(w.factions[f][1] for f in oc.get("winners", ()) if f < len(w.factions))
            outcome_text = (f"{'Victory' if win else 'Defeat'}: {win}" if win else
                            f"Episode ended ({oc.get('reason', '')})") + (f" - {oc.get('reason', '')}" if win else "")
        if w.done or out is not None:
            m = core.episode_metrics()
            metrics = dict(m["global"])
        return FrameState(
            step=int(core.env_steps if step is None else step), round=int(w.round),
            mode="combat" if w.mode == int(GameMode.COMBAT) else "exploration", active_id=int(w.active) if not w.done else -1,
            initiative=[int(e) for e in w.order], entities=ents, zones=zones, visible=visible, explored=explored,
            light_level=np.array(w.light_grid(), np.uint8), doors=doors, messages=msgs, rewards=rewards,
            returns=returns, relations=w.relations.astype(int).tolist(), objectives=objectives, decision=decision,
            done=bool(w.done), outcome=outcome_text, metrics=metrics, annotations=dict(annotations or {}))

    @staticmethod
    def event_dicts(events) -> list[dict]:
        return [e.to_dict() for e in events]


# ---------------------------------------------------------------------------------------------- action descriptions
def _area_cells(world, actor, ab, target_xy):
    from emberdeep.core import _geom_fallback as GF
    area = ab.area
    if area is None:
        return []
    shape = int(area.shape)
    size_sq = max(1, area.size_ft // 5)
    width_sq = max(1, area.width_ft // 5)
    tx, ty = target_xy
    from_caster = area.origin == "self" or shape == int(AreaShape.EMANATION)
    if shape == int(AreaShape.EMANATION):
        tx, ty = actor.x, actor.y
    try:
        cells = GF.geo.area_cells(shape, actor.x, actor.y, int(tx), int(ty), size_sq, width_sq, world.map.opaque,
                                  world.map.H, world.map.W, from_caster)
    except Exception:  # noqa: BLE001
        log.debug("area_cells failed", exc_info=True)
        return []
    return [(int(x), int(y)) for x, y in cells]


def describe_actions(core, agent: str, with_estimates: bool = True) -> list[ActionDescriptor]:
    """Descriptors for every currently legal action of ``agent`` (empty unless it is the decider)."""
    ctx = core.context(agent)
    lay = core.layout
    w = core.world
    eid = core.agent_eid[agent]
    c = w.entities[eid]
    out: list[ActionDescriptor] = []
    mask = ctx.action_mask
    abil = ctx.abilities if ctx.abilities is not None else w.abilities(eid)
    vocab = core.info.comm_vocab
    for a in np.flatnonzero(mask).tolist():
        d = lay.decode(a)
        kind = d[0]
        if kind == "end_turn":
            out.append(ActionDescriptor(a, "end_turn", "End turn" if ctx.is_decider else "Wait", icon="hourglass",
                                        cost="free", tooltip="End your turn."))
        elif kind == "move":
            dx, dy = DIR_VECTORS[d[1]]
            out.append(ActionDescriptor(a, "move", f"Move {_DIR_NAMES[d[1]]}", icon="boot-prints", cost="movement",
                                        target_cell=(c.x + dx, c.y + dy), direction=d[1]))
        elif kind in ("ascend", "descend"):
            out.append(ActionDescriptor(a, "move", kind.capitalize(), icon="wing-cloak", cost="movement",
                                        target_cell=(c.x, c.y)))
        elif kind == "ability":
            k, t = d[1], d[2]
            ab = abil[k]
            tid, tcell = -1, None
            if t < lay.E:
                tid = int(ctx.slot_eids[t])
                te = w.entities[tid]
                tcell = (te.x, te.y)
                label = ab.name if tid == eid else f"{ab.name} -> {te.name}"
            else:
                p = ctx.pts[t - lay.E]
                tcell = (int(p[0]), int(p[1]))
                label = f"{ab.name} -> ({tcell[0]}, {tcell[1]})"
            hit = dmg = None
            if with_estimates and tid >= 0 and tid != eid and w.entities[tid].kind == int(EntityKind.CREATURE):
                try:
                    hit, dmg = w.estimate(eid, k, tid)
                    hit, dmg = float(hit), float(dmg)
                except Exception:  # noqa: BLE001
                    hit = dmg = None
            area = _area_cells(w, c, ab, tcell) if (ab.area is not None and tcell is not None) else []
            out.append(ActionDescriptor(a, "ability", label, ability_id=ab.id, icon=ab.icon or "", cost=_COST_NAME.get(
                ab.cost, ""), target_eid=tid if tid != eid else -1, target_cell=tcell, area_cells=area,
                hit_chance=hit, expected_damage=dmg, tooltip=ab.name))
        elif kind == "approach":
            tid = int(ctx.slot_eids[d[1]])
            te = w.entities[tid]
            out.append(ActionDescriptor(a, "approach", f"Approach {te.name}", icon="walking-boot", cost="movement",
                                        target_eid=tid, target_cell=(te.x, te.y)))
        elif kind == "say":
            tok = d[1]
            txt = vocab[tok] if tok < len(vocab) else f"signal {tok}"
            out.append(ActionDescriptor(a, "say", f"Say: {txt}", icon="chat-bubble", cost="free"))
        else:
            dec = ctx.decision
            opt = dec.options[d[1]] if dec is not None and d[1] < len(dec.options) else None
            out.append(ActionDescriptor(a, "decision", (opt.label or opt.key) if opt is not None else f"Option {d[1]}",
                                        icon="swords-emblem", cost="reaction"))
    return out


# ---------------------------------------------------------------------------------------------- bridge
class RenderBridge:
    """Lazily creates the renderer and feeds it frames / events (``render_mode`` human or rgb_array)."""

    MAX_PENDING = 64

    def __init__(self, core, cfg):
        self.core = core
        self.cfg = cfg
        self.builder = FrameBuilder(core)
        self.renderer = None
        self.scene: StaticScene | None = None
        self.pending: list[tuple[FrameState, list[dict]]] = []
        self.annotations: dict = {}
        self.closed = False
        self._scene_dirty = True
        self._last_frame: FrameState | None = None
        self.reset()

    def reset(self) -> None:
        """New episode: new static scene and first frame (with the events produced while resetting)."""
        self.builder.reset()
        self.scene = self.builder.scene()
        self._scene_dirty = True
        self.pending = []
        ev = list(getattr(self.core, "reset_events", []))
        frame = self.builder.frame(ev, None, self.annotations)
        self._last_frame = frame
        self.pending.append((frame, self.builder.event_dicts(ev)))

    def after_step(self, out) -> None:
        frame = self.builder.frame(out.events, out, self.annotations)
        self._last_frame = frame
        self.pending.append((frame, self.builder.event_dicts(out.events)))
        if len(self.pending) > self.MAX_PENDING:
            del self.pending[0:len(self.pending) - self.MAX_PENDING]

    def set_annotations(self, annotations: dict) -> None:
        self.annotations = dict(annotations)
        if self._last_frame is not None:
            self._last_frame.annotations = dict(annotations)

    def _ensure_renderer(self):
        if self.renderer is None:
            from emberdeep.render import Renderer
            cfg = self.cfg
            self.renderer = Renderer(cfg.render_mode, cfg.render_width, cfg.render_height, cfg.render_quality,
                                     cfg.render_fps, cfg.render_hud, cfg.render_sound, cfg.render_pov,
                                     seed=int(self.core.world.seed))
            self.renderer.speed = float(cfg.render_speed)
            self._scene_dirty = True
        return self.renderer

    def render(self):
        if self.closed:
            return None
        r = self._ensure_renderer()
        if self._scene_dirty:
            r.set_scene(self.scene)
            self._scene_dirty = False
        if not r.is_open():
            return None
        pending, self.pending = self.pending, []
        for frame, events in pending:
            r.push(frame, events)
        if self.cfg.render_mode == "human":
            r.play_until_idle()
            return None
        r.play_until_idle(max_seconds=6.0)
        return r.capture()

    def request_human_action(self, agent: str) -> int:
        """Block until a human picks one of ``agent``'s legal actions in the renderer window."""
        r = self._ensure_renderer()
        if self._scene_dirty:
            r.set_scene(self.scene)
            self._scene_dirty = False
        pending, self.pending = self.pending, []
        for frame, events in pending:
            r.push(frame, events)
        r.play_until_idle()
        acts = describe_actions(self.core, agent)
        return int(r.request_human_action(agent, acts))

    def is_open(self) -> bool:
        """False once the user closed the window (or ``close()`` was called)."""
        return not self.closed and (self.renderer is None or self.renderer.is_open())

    def wait_until_closed(self) -> None:
        """Keep the window alive (final scoreboard etc.) until the user closes it."""
        r = self._ensure_renderer() if not self.closed else None
        if r is None or r.mode != "human":
            return
        self.render()
        while r.is_open():
            r.run_frame()

    def close(self) -> None:
        self.closed = True
        if self.renderer is not None:
            try:
                self.renderer.close()
            finally:
                self.renderer = None
