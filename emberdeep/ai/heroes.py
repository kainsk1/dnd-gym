"""Hero bots: one class-aware scripted player per SRD class (``hero_<class>``) plus the generic ``hero``.

Used as evaluation baselines, as scripted teammates for single-agent training, and as the "reasonable player" reference
for social-dilemma scenarios.  Combat behaviour is the shared utility AI with a class-tuned
:class:`~emberdeep.ai.profiles.Profile`; out of combat the bots

* heal wounded friends, vote for a short rest when hurt and safe, drink a potion when badly hurt and unable to rest,
* open doors (key, thieves' tools or brute force), pick up loot and open chests,
* walk to the regions / exits demanded by the world's objectives,
* otherwise explore the nearest frontier of the faction's map knowledge, the first hero (lowest eid) leading and the
  others following within a few squares.

No class feature is assumed to exist: abilities are found by id / tag (see ``ai/abinfo.py``).
"""
from __future__ import annotations

import numpy as np

from emberdeep.ai.abinfo import U
from emberdeep.ai.base import end_turn, jitter, mem, remember, turn_mem, turn_remember
from emberdeep.ai.explore import (approach_entity, closed_doors, door_toward, frontier_cells, interact_with, lootables,
                                  nearest_cell, objective_cells, update_seen, walk_to_cell)
from emberdeep.ai.policy import UtilityPolicy
from emberdeep.ai.profiles import ARCHER, BRUTE, CASTER, SKIRMISHER, SUPPORT, Profile
from emberdeep.ai.scene import Scene
from emberdeep.ai.tactics import cost_grids, mobile, reach_cost
from emberdeep.ai.utility import ab_act, survey
from emberdeep.core._geom_fallback import geo
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import DIR_VECTORS, EntityKind, GameMode

HERO_PROFILES = {
    "fighter": BRUTE.with_(name="hero_fighter", guard_w=0.8, safety=0.15, retreat_hp=0.25, heal_thr=0.35),
    "barbarian": BRUTE.with_(name="hero_barbarian", guard_w=0.4, safety=0.0, retreat_hp=0.12, stance_w=1.0),
    "paladin": BRUTE.with_(name="hero_paladin", guard_w=0.8, heal_w=1.0, heal_thr=0.4, safety=0.15, retreat_hp=0.25,
                           buff_w=1.0),
    "rogue": SKIRMISHER.with_(name="hero_rogue", band=None, flank_w=1.4, safety=0.45, retreat_hp=0.4, hide_w=0.5,
                              kite=False, isolated_w=0.3),
    "monk": SKIRMISHER.with_(name="hero_monk", band=None, flank_w=0.5, safety=0.3, retreat_hp=0.3, hide_w=0.0,
                             kite=False, isolated_w=0.0),
    "ranger": ARCHER.with_(name="hero_ranger", band=(4, 12), safety=0.7),
    "wizard": CASTER.with_(name="hero_wizard"),
    "sorcerer": CASTER.with_(name="hero_sorcerer"),
    "warlock": CASTER.with_(name="hero_warlock", band=(4, 14), slot_w=0.5),
    "cleric": SUPPORT.with_(name="hero_cleric", band=(1, 8), safety=0.5, kite=False, guard_w=0.3),
    "druid": SUPPORT.with_(name="hero_druid", band=(3, 12), safety=0.7),
    "bard": SUPPORT.with_(name="hero_bard", band=(2, 10), safety=0.7, heal_thr=0.5),
}
GENERIC = BRUTE.with_(name="hero", guard_w=0.4, safety=0.2, retreat_hp=0.25)

REST_MAX = 3                # short rests a bot asks for per episode
REST_HP = 0.7               # ask for a rest when any friend is below this hp fraction


class HeroPolicy(UtilityPolicy):
    """Class-aware hero bot.  ``cls`` is the class id ("" = look at the creature's levels)."""

    def __init__(self, cls: str = "", seed: int | None = None):
        self.cls = cls
        super().__init__(GENERIC, name=f"hero_{cls}" if cls else "hero", seed=seed)

    def profile_for(self, sc: Scene) -> Profile:
        cls = self.cls
        if not cls:
            from emberdeep.ai.registry import _pc_class
            cls = _pc_class(sc.c)
        return HERO_PROFILES.get(cls, GENERIC)

    # ------------------------------------------------------------------------------------------ out of combat
    def idle(self, sc: Scene, prof: Profile) -> Act:
        c, w = sc.c, sc.w
        if w.mode == int(GameMode.COMBAT) and not sc.standing:
            return end_turn()
        sv = None
        # 1. patch up friends (spells, potions, features)
        if sc.allies or sc.downed_allies or sc.hpf < 0.9:
            sv = survey(sc, prof.with_(heal_thr=max(prof.heal_thr, 0.7)))
            heal = [x for x in sv.cands if x.tag.startswith(("heal", "stab"))]
            if heal:
                best = max(heal, key=lambda x: x.score)
                if best.score > 0.3:
                    return best.act
        a = self.rest_logic(sc)
        if a is not None:
            return a
        if mem(c).get("rest_wait", -9) >= w.round - 1 and c.rest_vote:
            return end_turn()
        if sc.hpf < 0.4 and sc.mask()[U["drink_potion"]] and not sc.standing:
            return ab_act(U["drink_potion"])
        a = self.explore(sc, prof)
        if a is not None:
            return a
        return end_turn()

    # ------------------------------------------------------------------------------------------ short rests
    def rest_logic(self, sc: Scene) -> Act | None:
        c, w = sc.c, sc.w
        if w.mode != int(GameMode.EXPLORATION) or sc.standing:
            if c.rest_vote and sc.standing:
                return ab_act(U["rest"]) if sc.mask()[U["rest"]] else None
            return None
        m = mem(c)
        mates = [a for a in sc.allies if a.conscious] + [c]
        hurt = min(x.hp / max(1, x.max_hp) for x in mates) < REST_HP and not sc.downed_allies
        used = m.get("rests", 0)
        if c.rest_vote:
            # waited long enough for the others? withdraw the vote and carry on
            if w.round - m.get("rest_round", w.round) >= 3:
                if sc.mask()[U["rest"]]:
                    return ab_act(U["rest"])
            remember(c, rest_wait=w.round)
            return None
        if hurt and used < REST_MAX and sc.mask()[U["rest"]] and not any(
                x.hp < x.max_hp * 0.2 and x.status & (S.DYING) for x in mates):
            remember(c, rest_round=w.round, rest_wait=w.round, rests=used + 1)
            return ab_act(U["rest"])
        return None

    # ------------------------------------------------------------------------------------------ exploration
    def explore(self, sc: Scene, prof: Profile) -> Act | None:
        """Choose (and keep) an exploration goal: loot / objective entities, objective regions, frontier, doors."""
        c, w = sc.c, sc.w
        if not mobile(sc) and c.ts.action_used:
            return None
        update_seen(sc)
        a = self.escape_act(sc)
        if a is not None:
            return a
        vip = [x for x in sc.allies if "vip" in x.tags and x.conscious]
        if vip:                                          # escorting: never run ahead of the VIP
            v = min(vip, key=lambda x: sc.dist2(c, x))
            d = sc.dist2(c, v)
            if d > 12 and mobile(sc):
                a = approach_entity(sc, v)
                if a is not None:
                    return a
            if d > 6:
                return None                      # wait for the slow VIP instead of dithering back and forth
            from emberdeep.ai.vip import goal_cells
            gset = set(goal_cells(w))
            if gset and mobile(sc) and d <= 6 and not (v.x, v.y) in gset and any(p in gset for p in c.cells()):
                mm = w.move_mask(sc.eid)                 # do not block the VIP's way into the goal area
                best, bs = None, (-1, -1)
                for dd in range(8):
                    if mm[dd]:
                        dx, dy = DIR_VECTORS[dd]
                        nx, ny = c.x + dx, c.y + dy
                        nd = max(abs(nx - v.x), abs(ny - v.y))
                        key = (1 if (nx, ny) not in gset else 0, nd)
                        if nd >= d and key > bs:
                            best, bs = dd, key
                if best is not None:
                    return Act(ActKind.MOVE, best)
        # the party leader (a lock-picker if there is one, else the lowest eid) explores; the others stay close
        mates = [x for x in sc.allies if x.present and x.conscious and x.is_pc]
        leader = min(mates + [c], key=lambda x: (x.find_item("thieves_tools") is None, x.eid))
        if leader is not c:
            d = sc.dist2(c, leader)
            if d > 4 and mobile(sc):
                a = approach_entity(sc, leader)
                if a is not None:
                    return a
            elif d <= 1 and mem(leader).get("stuck", 0) >= 1 and mobile(sc):
                a = self._make_way(sc, leader)
                if a is not None:
                    return a
            return None
        # keep pursuing the current goal until reached or given up (prevents dithering between goals)
        goal = mem(c).get("goal")
        if goal is not None and goal[-1] >= w.round:
            a = self._pursue(sc, goal)
            if a is not None:
                remember(c, stuck=0)
                return a
            if goal[0] == "wait":
                return None
        goal = self._choose_goal(sc)
        if goal is None:
            remember(c, goal=None, stuck=mem(c).get("stuck", 0) + 1)
            return None
        remember(c, goal=goal)
        a = self._pursue(sc, goal)
        if a is None:
            remember(c, goal=None)
            return None
        remember(c, stuck=0)
        return a

    def hunt(self, sc: Scene, prof: Profile, gate):
        if any("vip" in a.tags for a in sc.allies):
            return None      # escorting: the mission is the VIP, not ghost-chasing a bandit
        return super().hunt(sc, prof, gate)

    def fight(self, sc: Scene, prof: Profile, gate):
        """Heroes carrying the prize run for the exit even with enemies around (grab and go)."""
        g0 = mem(sc.c).get("gold0")
        guard_up = any('guardian' in f.tags and not f.cond_mask >> 13 & 1 for f in sc.foes)
        if g0 is not None and (sc.c.gold >= g0 + 100 or guard_up) and sc.c.ts.movement_left >= 5:
            a = self.escape_act(sc)
            if a is not None:
                return a
        return super().fight(sc, prof, gate)

    # ------------------------------------------------------------------------------------------ leaving the map
    def wants_escape(self, sc: Scene) -> bool:
        """Should the party leave through an exit now?  (payoff / objectives say escaping matters and the job is done.)"""
        from emberdeep.core.types import ObjectiveKind
        c, w = sc.c, sc.w
        rt = getattr(w, "scenario_runtime", None)
        spec = getattr(rt, "spec", None)
        payoff = spec.rewards.terminal_payoff if spec is not None else "none"
        esc_obj = [st for st in w.objectives if st.spec.kind == ObjectiveKind.ESCAPE and not st.done
                   and (not st.faction_idx or c.faction in st.faction_idx)]
        cfg = getattr(rt, "cfg", {}) if rt is not None else {}
        if not (esc_obj or "escaped" in payoff or cfg.get("relic_escape")):
            return False
        g0 = mem(c).get("gold0")
        if g0 is None:
            remember(c, gold0=c.gold)
            g0 = c.gold
        guard_up = any('guardian' in f.tags and not f.cond_mask >> 13 & 1 for f in sc.foes)
        laden = (c.gold >= g0 + 100 or guard_up) and "escaped" in payoff   # grab and go: run with the loot
        if not laden and any(not f.cond_mask >> 13 & 1 and ((sc.dist(f) <= 8 and (f in sc.vis or f.eid in c.aware)) or (f in sc.vis and sc.dist(f) <= 12 and not any(
                t == 'stationary' or t.startswith('guard:') for t in f.tags))) for f in sc.standing):
            return False                                   # awake hostiles around: deal with them first
        mates = [x for x in sc.allies if x.is_pc and x.conscious] + [c]
        avg = sum(x.hp / max(1, x.max_hp) for x in mates) / len(mates)
        late = w.round >= 0.45 * w.rules.max_rounds
        open_obj = [st for st in w.objectives if not st.done and not st.failed and st.spec.kind not in (
            ObjectiveKind.ESCAPE, ObjectiveKind.PROTECT, ObjectiveKind.SURVIVE)]
        if open_obj and not late and avg > 0.35:
            return False
        if not laden and any(o.eid not in self._skips(sc) for o in lootables(w) + self.treasure_altars(sc)) and not late and avg > 0.35 and not esc_obj:
            return False
        boss = [x for x in w.creatures if "boss" in x.tags and x.present and not x.status & S.DEAD
                and x.faction != c.faction and not x.status & S.ESCAPED]
        if boss and not late and avg > 0.35 and not esc_obj:
            return False
        return True

    @staticmethod
    def treasure_altars(sc: Scene) -> list:
        """Altars that pay out gold when used (cursed coffer, hoard pedestals)."""
        return [o for o in sc.w.objects if o.present and o.kind == EntityKind.ALTAR and any(
            k in (o.hook or "") for k in ("coffer", "treasure", "hoard", "donat")) and "donat" not in (o.hook or "")]

    def escape_act(self, sc: Scene) -> Act | None:
        c, w = sc.c, sc.w
        if not self.wants_escape(sc):
            return None
        exits = [o for o in w.objects if o.kind == EntityKind.EXIT and o.present and o.eid not in self._skips(sc)]
        if not exits:                                  # sealed exit: crank the winch / pull the lever that opens it
            exits = [o for o in w.objects if o.present and o.eid not in self._skips(sc) and (
                (o.kind == EntityKind.ALTAR and "winch" in (o.hook or "")) or o.kind == EntityKind.LEVER)]
        if not exits:
            return None
        if not mobile(sc) and w.distance(sc.eid, min(exits, key=lambda o: w.distance(sc.eid, o.eid)).eid) > 1:
            return None
        e = min(exits, key=lambda o: w.distance(sc.eid, o.eid))
        if w.distance(sc.eid, e.eid) <= 1:
            a = interact_with(sc, e)
            if a is None and not c.ts.free_used:
                self._skip(sc, e.eid, 10)
            return a
        a = approach_entity(sc, e)
        if a is None:
            a = door_toward(sc, e.x, e.y)            # a closed door is in the way
        if a is None:
            self._skip(sc, e.eid, 5)
        return a

    # --- goals are tuples: ("obj", eid, until) | ("cell", x, y, until)
    def _choose_goal(self, sc: Scene):
        c, w = sc.c, sc.w
        skip = self._skips(sc)
        cost_ap, cost_wk = cost_grids(sc)
        rule = 1 if w.rules.diagonal_rule == "5-10-5" else 0
        dist_ap = geo.dijkstra(cost_ap, c.x, c.y, 1e6, rule)
        cells, objs = objective_cells(w, c)
        near = []
        if sc.hpf < 0.7:
            for o in w.objects:
                if o.present and o.kind == EntityKind.ALTAR and ("heal" in (o.hook or "") or "fount" in (o.hook or "")):
                    near.append(o)
        best = None
        g0 = mem(c).get("gold0", c.gold)
        greedy = [] if c.gold >= g0 + 100 else self.treasure_altars(sc)
        for o in lootables(w) + greedy + objs + near:
            if o.eid in skip:
                continue
            d = reach_cost(dist_ap, o, 1)
            if d < (20.0 if any('vip' in x.tags for x in sc.allies) else 60.0) and (best is None or d < best[0]):
                best = (d, o)
        if best is not None:
            return ("obj", best[1].eid, w.round + 25)
        if cells:
            if {(c.x, c.y)} & set(cells):
                return ("wait", w.round + 3)
            tgt = nearest_cell(dist_ap, cells)
            if tgt is not None:
                return ("cell", tgt[0], tgt[1], w.round + 25)
        seen = mem(c).get("seen")
        visited = mem(c).get("visited") or {}
        fr = frontier_cells(sc, seen, dist_ap)
        if fr.any():
            ys, xs = np.nonzero(fr)
            ok = np.array([visited.get((int(x), int(y)), -1) < w.round for x, y in zip(xs, ys)], bool)
            if ok.any():
                xs, ys = xs[ok], ys[ok]
                d = dist_ap[ys, xs] / 5.0
                mates = [x for x in sc.allies if x.present and x.conscious and x.is_pc]
                if mates:
                    gx = float(np.mean([x.x for x in mates]))
                    gy = float(np.mean([x.y for x in mates]))
                    d = d + 0.3 * np.maximum(np.abs(xs - gx), np.abs(ys - gy))
                order = np.lexsort((xs + 1000 * ys, d))
                i = int(order[0])
                return ("cell", int(xs[i]), int(ys[i]), w.round + 25)
        doors = [o for o in closed_doors(w) if o.eid not in skip]
        best = None
        for o in doors:
            d = reach_cost(dist_ap, o, 1)
            if d < float("inf") and (best is None or d < best[0]):
                best = (d, o)
        if best is not None:
            return ("obj", best[1].eid, w.round + 25)
        return None

    def _pursue(self, sc: Scene, goal):
        c, w = sc.c, sc.w
        if goal[0] == "wait":
            return None
        if goal[0] == "obj":
            o = w.entities[goal[1]]
            if not o.present:
                return None
            if o.kind == EntityKind.DOOR and o.state.name in ("OPEN", "BROKEN"):
                return None
            if o.kind == EntityKind.CHEST and o.open:
                return None
            if o.kind == EntityKind.LOOT and not (o.gold or o.items):
                return None
            if w.distance(sc.eid, o.eid) <= 1:
                a = interact_with(sc, o)
                if a is None and not c.ts.action_used:
                    self._skip(sc, o.eid, 40)
                return a
            if not mobile(sc):
                return Act(ActKind.END_TURN) if False else None
            a = approach_entity(sc, o)
            if a is None:
                a = door_toward(sc, o.x, o.y)
            if a is None:
                self._skip(sc, o.eid, 6)
            return a
        x, y = goal[1], goal[2]
        if (c.x, c.y) == (x, y):
            v = dict(mem(c).get("visited") or {})
            v[(x, y)] = w.round + 40
            remember(c, visited=v)
            return None
        if not mobile(sc):
            return None
        a = walk_to_cell(sc, x, y)
        if a is None:
            v = dict(mem(c).get("visited") or {})
            v[(x, y)] = w.round + 20
            remember(c, visited=v)
        return a

    @staticmethod
    def _skips(sc: Scene) -> frozenset:
        """Objects given up on recently (entries expire: doors may open later, paths may clear)."""
        d = mem(sc.c).get("skip") or {}
        return frozenset(e for e, until in d.items() if until >= sc.w.round)

    @staticmethod
    def _skip(sc: Scene, eid: int, rounds: int = 12) -> None:
        d = dict(mem(sc.c).get("skip") or {})
        d[eid] = sc.w.round + rounds
        remember(sc.c, skip=d)

    def _make_way(self, sc: Scene, leader) -> Act | None:
        """Step to a free square away from a stuck leader (it may need to walk through our square)."""
        c = sc.c
        mm = sc.w.move_mask(sc.eid)
        best = None
        for d in range(8):
            if not mm[d]:
                continue
            dx, dy = DIR_VECTORS[d]
            nd = max(abs(c.x + dx - leader.x), abs(c.y + dy - leader.y))
            if nd >= 2 and (best is None or nd < best[0]):
                best = (nd, d)
        return None if best is None else Act(ActKind.MOVE, best[1])

    @staticmethod
    def _skips(sc: Scene) -> frozenset:
        """Objects given up on recently (entries expire after 12 rounds: doors may open later, paths may clear)."""
        d = mem(sc.c).get("skip") or {}
        return frozenset(e for e, until in d.items() if until >= sc.w.round)

    @staticmethod
    def _skip(sc: Scene, eid: int, rounds: int = 12) -> None:
        d = dict(mem(sc.c).get("skip") or {})
        d[eid] = sc.w.round + rounds
        remember(sc.c, skip=d)

    def _go_to_object(self, sc: Scene, objs: list, dist_ap, skip, limit: float) -> Act | None:
        """Walk to / use the cheapest reachable object; unusable ones are remembered and ignored."""
        c, w = sc.c, sc.w
        cand = []
        for o in objs:
            d = reach_cost(dist_ap, o, 1)
            if d < limit:
                cand.append((d, o))
        cand.sort(key=lambda t: t[0])
        for d, o in cand[:3]:
            if w.distance(sc.eid, o.eid) <= 1:
                a = interact_with(sc, o)
                if a is not None:
                    return a
                if not c.ts.action_used:
                    self._skip(sc, o.eid, 40)
                continue
            if mobile(sc):
                a = approach_entity(sc, o)
                if a is not None:
                    return a
                self._skip(sc, o.eid, 6)
            else:
                return None
        return None


    def goal(self, sc: Scene, prof: Profile):
        """Escorts hug the VIP: cells far from it are penalised."""
        vip = [x for x in sc.allies if "vip" in x.tags and x.conscious]
        if not vip or sc.downed_allies:
            return None
        v = min(vip, key=lambda x: sc.dist2(sc.c, x))
        vx, vy = v.x, v.y
        return lambda cx, cy: -0.6 * np.maximum(0, np.maximum(np.abs(cx - vx), np.abs(cy - vy)) - 3)
