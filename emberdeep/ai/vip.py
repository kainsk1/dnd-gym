"""VIP escort NPC policy: follow the nearest escort, walk to the exit, hide / dodge when enemies are close."""
from __future__ import annotations

from emberdeep.ai.abinfo import U
from emberdeep.ai.base import end_turn
from emberdeep.ai.explore import approach_entity, nearest_cell, walk_to_cell
from emberdeep.ai.policy import UtilityPolicy
from emberdeep.ai.profiles import VIP
from emberdeep.ai.tactics import cost_grids, hide_ability, mobile
from emberdeep.ai.utility import ab_act
from emberdeep.core._geom_fallback import geo
from emberdeep.core.entities import S
from emberdeep.core.types import EntityKind


def goal_cells(w) -> list:
    """Cells of the escort goal region (scenario runtime config, else the 'exit' region)."""
    rt = getattr(w, "scenario_runtime", None)
    name = "exit"
    if rt is not None:
        esc = getattr(rt, "cfg", {}).get("escort") if isinstance(getattr(rt, "cfg", None), dict) else None
        if esc:
            name = esc.get("region", name)
    return [tuple(p) for p in w.regions.get(name, ())]


class VipPolicy(UtilityPolicy):
    def __init__(self, seed=None):
        super().__init__(VIP, name="vip", seed=seed)

    def hunt(self, sc, prof, gate):
        return None          # an escort target never goes looking for trouble

    def fight(self, sc, prof, gate):
        w = sc.w
        near = [f for f in sc.vis if sc.dist(f) <= 7]
        if not near or not any(a for a in sc.allies if "vip" not in a.tags and a.conscious):
            return self.idle(sc, prof)       # nobody left to protect us: make a run for the goal
        adj = [f for f in near if sc.dist(f) <= 1]
        mask = sc.mask()
        escorts = [a for a in sc.allies if "vip" not in a.tags and a.conscious]
        if escorts and mobile(sc):                      # retreat to the escorts and let them fight
            e = min(escorts, key=lambda a: sc.dist2(sc.c, a))
            if sc.dist2(sc.c, e) > 2:
                a = approach_entity(sc, e)
                if a is not None:
                    return a
        if adj:
            if mask[U["dodge"]]:
                return ab_act(U["dodge"])
        else:
            h = hide_ability(sc)
            if h is not None and not sc.c.status & S.HIDDEN:
                return ab_act(h[0])
        return super().fight(sc, prof, gate)

    def idle(self, sc, prof):
        c, w = sc.c, sc.w
        escorts = [a for a in sc.allies if "vip" not in a.tags and a.conscious]
        if escorts and sc.standing and any(sc.dist(f) <= 7 for f in sc.vis):
            return end_turn()
        cells = goal_cells(w)
        exits = [o for o in w.objects if o.kind == EntityKind.EXIT and o.present]
        if escorts:
            near = min(escorts, key=lambda a: sc.dist2(c, a))
            d = sc.dist2(c, near)
            if d > 12 and mobile(sc):
                a = approach_entity(sc, near)
                if a is not None:
                    return a
            if d > 16 or not cells:
                return end_turn()
        if mobile(sc):
            if cells and (c.x, c.y) not in set(cells):
                cost_ap, _ = cost_grids(sc)
                dist = geo.dijkstra(cost_ap, c.x, c.y, 1e6, 0)
                tgt = nearest_cell(dist, cells)
                if tgt is not None:
                    a = walk_to_cell(sc, tgt[0], tgt[1])
                    if a is not None:
                        return a
            elif exits:
                e = min(exits, key=lambda o: sc.w.distance(sc.eid, o.eid))
                if sc.w.distance(sc.eid, e.eid) <= 1:
                    k = U["interact"]
                    if sc.mask()[k] and sc.w.abilities(sc.eid)[k].valid_target(sc.w, c, e.eid):
                        return ab_act(k, e.eid)
                a = approach_entity(sc, e)
                if a is not None:
                    return a
        return end_turn()
