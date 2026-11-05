"""``UtilityPolicy``: the shared scripted AI.  One class, many :class:`~emberdeep.ai.profiles.Profile` parameter sets.

Turn structure (every call to :meth:`act` returns exactly one atomic act):

1. **Gate** -- ``asleep`` / ``guard:<region>`` / ``stationary`` spawn tags restrict what the creature may do.
2. **Social layer** (``ai/social.py``) -- truce replies, morale checks, fleeing, surrender.
3. **Situational** -- stand up when prone, idle / patrol / hunt when no foe is known.
4. **Survey** (``ai/utility.py``) -- score every currently legal (ability, target / point).
5. **Movement** (``ai/tactics.py``) -- approach, kite, cover, flank, retreat, hide, disengage; gain-scored.
6. The best candidate wins; below a small threshold the turn ends.
"""
from __future__ import annotations

import numpy as np

from emberdeep.ai.abinfo import U, info
from emberdeep.ai.base import Policy, end_turn, jitter, mem, remember, turn_mem, turn_remember
from emberdeep.ai.profiles import Profile
from emberdeep.ai.scene import Scene
from emberdeep.ai import social, tactics
from emberdeep.ai.utility import Cand, Survey, ab_act, survey
from emberdeep.core import resolution as R
from emberdeep.core._geom_fallback import geo
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.types import DIR_VECTORS, EntityKind, GameMode, Relation, TargetType

_DIR_INDEX = {v: i for i, v in enumerate(DIR_VECTORS)}
MAX_ACTS_PER_TURN = 48
TRACE: list | None = None     # debugging: set to [] to record (round, eid, name, chosen tag, score, top candidates)
THRESHOLD = 0.2


class UtilityPolicy(Policy):
    """Profile-driven utility AI."""

    def __init__(self, profile: Profile, name: str | None = None, seed: int | None = None):
        self.profile = profile
        self.name = name or profile.name
        self.seed = seed

    # ------------------------------------------------------------------------------------------ public API
    def act(self, world, eid: int) -> Act:
        c = world.entities[eid]
        tm = turn_mem(world, c)
        pm = tm.get("path")
        if pm is not None and pm[0] and pm[1] == c.hp and world.last_step_valid and tm.get("last", (0,))[0] == 1 \
                and c.ts.movement_left >= 5 and not c.cond_mask >> 3 & 1 and TRACE is None:
            # fast path: keep walking the route planned earlier this turn (nothing hit us since)
            nx, ny = pm[0][0]
            d = _DIR_INDEX.get((nx - c.x, ny - c.y))
            if d is not None and world.move_mask(eid)[d]:
                turn_remember(world, c, path=(pm[0][1:], c.hp), n=tm.get("n", 0) + 1, last=(1, d, -1, None))
                return Act(ActKind.MOVE, d)
        if not tm or tm.get("last", (0,))[0] in (1, 2, 3, 5):      # first act of the turn, or we just moved
            self.refresh_awareness(world, c)
        sc = Scene(world, c)
        n = tm.get("n", 0) + 1
        if n > MAX_ACTS_PER_TURN:
            return end_turn()
        bad = tm.get("bad", frozenset())
        last = tm.get("last")
        if last is not None and not world.last_step_valid:
            bad = bad | {last}              # never repeat an act the engine just rejected
        n0 = len(TRACE) if TRACE is not None else 0
        try:
            a = self._act(world, c, sc)
        except Exception:                                   # noqa: BLE001 - a bot must never crash an episode
            from emberdeep.ai.safe import safe_fallback
            a = safe_fallback(world, c, sc)
        key = (int(a.kind), a.arg, a.target, a.point)
        if key in bad:
            a = end_turn()
            key = (0, -1, -1, None)
        turn_remember(world, c, n=n, last=key, bad=bad)
        if TRACE is not None and len(TRACE) == n0:
            TRACE.append((world.round, eid, c.name, f"other:{a.kind.name}", 0.0, [("mode", world.mode)]))
        return a

    @staticmethod
    def refresh_awareness(world, c) -> None:
        """Update ``c.aware`` from sight and from what allies see (scripted creatures call out sightings)."""
        if not c.present or not c.conscious:
            return
        world.update_awareness(c)
        rel = world.relations[c.faction]
        for o in world.creatures:
            if o is c or not o.present or o.faction != c.faction or not o.conscious:
                continue
            for e, v in o.aware.items():
                if e in c.aware:
                    continue
                f = world.entities[e]
                if f.present and f.faction != c.faction and int(rel[f.faction]) == 0 and not f.status & S.DEAD \
                        and world.sees(o.eid, e):
                    c.aware[e] = (f.x, f.y, world.round)

    # ------------------------------------------------------------------------------------------ hooks for subclasses
    def profile_for(self, sc: Scene) -> Profile:
        return self.profile

    def idle(self, sc: Scene, prof: Profile) -> Act:
        """Nothing hostile is known: exploration behaviour.  Monsters wait (or patrol)."""
        return self.default_idle(sc, prof)

    def extra_candidates(self, sc: Scene, prof: Profile, sv: Survey) -> list[Cand]:
        """Profile / class specific extra candidates (hero bots add loot, doors ...)."""
        return []

    def urgency(self, sc: Scene, prof: Profile) -> float:
        if prof.retreat_hp <= 0 or not sc.standing:
            return 0.0
        hpf = sc.hpf
        if hpf >= prof.retreat_hp:
            return 0.0
        return min(1.0, (prof.retreat_hp - hpf) / max(0.05, prof.retreat_hp))

    # ------------------------------------------------------------------------------------------ the act pipeline
    def _act(self, world, c, sc: Scene) -> Act:
        prof = self.profile_for(sc)
        gate = self.gate(sc, prof)
        if gate == "sleep":
            return end_turn()
        a = social.pre_turn(sc, prof)
        if a is not None:
            return a
        if prof.style == "idle":
            return end_turn()
        if not sc.standing and sc.downed_foes and prof.finish_downed > 0 and gate != "hold":
            a = self.finish_downed(sc, prof)
            if a is not None:
                return a
        if not sc.standing or (not sc.vis and not sc.known_foes()):
            if not sc.vis and sc.standing and world.mode == int(GameMode.COMBAT):
                h = self.hunt(sc, prof, gate)
                if h is not None:
                    return h
            return self.idle(sc, prof)
        a = self.fight(sc, prof, gate)
        if a.kind == ActKind.END_TURN and gate is None and not any(sc.dist(f) <= 10 for f in sc.vis):
            return self.idle(sc, prof)           # foes are only remembered / out of reach: carry on with other business
        return a

    # ------------------------------------------------------------------------------------------ gating by spawn tags
    def gate(self, sc: Scene, prof: Profile) -> str | None:
        """'sleep' (do nothing), 'hold' (no movement), or None.  Implements asleep / guard / stationary tags."""
        c, w = sc.c, sc.w
        tags = c.tags
        if not tags:
            return None
        m = mem(c)
        if "asleep" in tags and not m.get("awake"):
            wake = c.hp < c.max_hp or c.n_attacks_taken > 0
            if not wake:
                hr = w.rules.hearing_radius
                for f in sc.foes:
                    a = c.aware.get(f.eid)
                    if a is not None and w.distance(sc.eid, f.eid) <= hr // 2:
                        wake = True
                        break
                if not wake:
                    for al in sc.allies:
                        if al.hp < al.max_hp and w.distance(sc.eid, al.eid) <= hr:
                            wake = True
                            break
            if not wake:
                return "sleep"
            remember(c, awake=True)
        hold = False
        for t in tags:
            if t == "stationary":
                hold = True
            elif t.startswith("guard:") and not m.get("guard_on"):
                cells = w.regions.get(t[6:], ())
                trig = c.hp < c.max_hp or c.n_attacks_taken > 0
                if not trig and cells:
                    cs = set(map(tuple, cells))
                    for f in sc.foes:
                        if (f.x, f.y) in cs:
                            trig = True
                            break
                if trig:
                    remember(c, guard_on=True)
                else:
                    hold = True
        return "hold" if hold else None

    # ------------------------------------------------------------------------------------------ main combat decision
    def fight(self, sc: Scene, prof: Profile, gate: str | None) -> Act:
        c, w = sc.c, sc.w
        allow_move = gate != "hold"
        tm = turn_mem(w, c)
        # ambushers wait in hiding until prey comes close
        if prof.style == "ambush":
            a = self.ambush(sc, prof, allow_move)
            if a is not None:
                return a
        sig = (c.x, c.y, tuple((f.eid, f.x, f.y) for f in sc.vis))
        cc = tm.get("cc")
        if cc is not None and cc[0] == sig and tm.get("last", (0,))[0] == 4 and w.last_step_valid:
            a = self.reuse_attack(sc, cc[1])         # e.g. the second attack of a multiattack: nothing moved
            if a is not None:
                return a
        sv = Survey() if self.survey_pointless(sc) else survey(sc, prof)
        urg = self.urgency(sc, prof)
        cands = list(sv.cands)
        best_now = max(cands, key=lambda x: x.score) if cands else None
        # stand up from prone
        if c.cond_mask >> 10 & 1:
            k = U["stand_up"]
            if sc.mask()[k]:
                cands.append(Cand(4.5 if prof.band is None else 2.5, ab_act(k), "stand"))
        cands.extend(tactics.plan_move(sc, prof, sv, best_now, urg, allow_move, self.goal(sc, prof)))
        # dodge when hurt and cornered
        if urg > 0.4 and not c.ts.action_used:
            adj = [f for f in sc.standing if w.distance(sc.eid, f.eid) <= max(1, sc.offense(f)["reach"])]
            if adj and sc.mask()[U["dodge"]]:
                inc = sum(sc.offense(f)["m"] for f in adj)
                cands.append(Cand(0.3 * inc * (1 + urg), ab_act(U["dodge"]), "dodge"))
        cands.extend(self.extra_candidates(sc, prof, sv))
        bad = tm.get("bad", frozenset())
        last = tm.get("last")
        if last is not None and not w.last_step_valid:
            bad = bad | {last}
        if bad:
            cands = [x for x in cands if (int(x.act.kind), x.act.arg, x.act.target, x.act.point) not in bad]
        if (not cands or max(x.score for x in cands) < THRESHOLD) and allow_move and sc.known_foes():
            a = self.open_way(sc)
            if a is not None:
                return a
        if not cands:
            return end_turn()
        best = max(cands, key=lambda x: x.score)
        if cands and best.tag.startswith("atk"):
            top = sorted((x for x in cands if x.tag.startswith("atk")), key=lambda x: -x.score)[:3]
            turn_remember(w, c, cc=(sig, tuple((x.score, x.act.arg, x.act.target, x.tag) for x in top)))
        if TRACE is not None:
            TRACE.append((w.round, sc.eid, c.name, best.tag, round(best.score, 2),
                          [(x.tag, round(x.score, 2)) for x in sorted(cands, key=lambda x: -x.score)[:4]]))
        if best.score < THRESHOLD:
            return end_turn()
        t = best.tag
        if t.startswith(("atk", "aoe", "fin")):
            remember(c, last_atk=best.score)
        if t.startswith(("ctl", "buff", "dbf", "mark", "summon", "aoe")) and best.act.kind == ActKind.ABILITY:
            ab = w.abilities(sc.eid)[best.act.arg]
            if info(ab).conc:
                remember(c, conc_val=max(best.score, 3.0))
        return best.act

    def finish_downed(self, sc: Scene, prof: Profile) -> Act | None:
        """Only helpless (dying / stable) foes are left: walk up to them and finish them off."""
        sv = survey(sc, prof)
        fin = [x for x in sv.cands if x.tag.startswith("fin")]
        if fin:
            return max(fin, key=lambda x: x.score).act
        from emberdeep.ai.explore import approach_entity
        vis = [f for f in sc.downed_foes if sc.w.sees(sc.eid, f.eid)]
        if vis and tactics.mobile(sc):
            return approach_entity(sc, min(vis, key=lambda f: sc.dist(f)))
        return None

    def open_way(self, sc: Scene) -> Act | None:
        """Foes are known but nothing useful can be done: open the door that leads towards the nearest one."""
        from emberdeep.ai.explore import door_toward
        kf = sc.known_foes()
        c = sc.c
        f = min(kf, key=lambda k: max(abs(k[1] - c.x), abs(k[2] - c.y)))
        return door_toward(sc, f[1], f[2])

    @staticmethod
    def reuse_attack(sc: Scene, top):
        """Next best attack from the previous survey of this turn if it is still legal (positions unchanged)."""
        w, c = sc.w, sc.c
        abil = w.abilities(sc.eid)
        mask = None
        for score, k, tgt, tag in top:
            t = w.entities[tgt]
            if score < THRESHOLD or not t.present or t.status & (S.DEAD | S.DYING | S.STABLE | S.SURRENDERED):
                continue
            if k >= len(abil):
                continue
            ab = abil[k]
            if not ab.usable(w, c) or not ab.valid_target(w, c, tgt):
                continue
            return Act(ActKind.ABILITY, k, tgt)
        return None

    @staticmethod
    def survey_pointless(sc: Scene) -> bool:
        """The action is spent and no bonus / free ability that the survey scores is left: skip the ability survey."""
        c = sc.c
        ts = c.ts
        if not ts.action_used or ts.extra_actions or ts.attacks_left > 0 or ts.mseq:
            return False
        m = mem(c)
        v = m.get("nbs")
        if v is None or v[0] != c.version:
            from emberdeep.ai.abinfo import info as _info
            from emberdeep.ai.utility import _HANDLERS
            from emberdeep.core.abilities import N_UNIVERSAL
            from emberdeep.core.types import ActionCost
            abil = sc.w.abilities(sc.eid)
            need = any(ab.cost in (ActionCost.BONUS_ACTION, ActionCost.FREE) and _info(ab).role in _HANDLERS
                       for ab in abil[N_UNIVERSAL:])
            v = (c.version, need)
            remember(c, nbs=v)
        return not v[1]

    def goal(self, sc: Scene, prof: Profile):
        """Optional extra cell-score term ``f(cx, cy) -> array`` (subclasses)."""
        return None

    # ------------------------------------------------------------------------------------------ ambush
    def ambush(self, sc: Scene, prof: Profile, allow_move: bool) -> Act | None:
        c, w = sc.c, sc.w
        m = mem(c)
        hidden = bool(c.status & S.HIDDEN)
        near = min((w.distance(sc.eid, f.eid) for f in sc.vis), default=99)
        strike = 1 + c.ts.movement_left // 5
        waited = m.get("waited", 0)
        if hidden:
            if near <= strike or waited >= prof.patient:
                return None
            if c.ts.action_used or True:
                remember(c, waited=waited + 1)
                return end_turn()
        else:
            if near > 1 and waited < prof.patient and sc.mask()[U["hide"]] and not tactics._adjacent_foe(sc):
                remember(c, waited=waited + 1)
                return ab_act(U["hide"])
            hid = tactics.hide_ability(sc)
            if hid is not None and near > 1 and waited < prof.patient:
                remember(c, waited=waited + 1)
                return ab_act(hid[0])
        return None

    # ------------------------------------------------------------------------------------------ idle / hunt
    def default_idle(self, sc: Scene, prof: Profile) -> Act:
        c, w = sc.c, sc.w
        # healers still patch up wounded friends; summons follow their master
        if c.summoner >= 0 and 0 <= c.summoner < len(w.entities):
            m = w.entities[c.summoner]
            if m.present and w.distance(sc.eid, m.eid) > 3 and tactics.mobile(sc) and \
                    w.approach_mask(sc.eid, [m.eid])[0]:
                return Act(ActKind.APPROACH, -1, m.eid)
        if sc.allies or sc.downed_allies:
            sv = survey(sc, prof)
            heal = [x for x in sv.cands if x.tag.startswith(("heal", "stab"))]
            if heal:
                return max(heal, key=lambda x: x.score).act
        if "patrol" in c.tags and tactics.mobile(sc):
            a = self.patrol(sc)
            if a is not None:
                return a
        return end_turn()

    def patrol(self, sc: Scene) -> Act | None:
        c, w = sc.c, sc.w
        m = mem(c)
        home = m.get("home")
        if home is None:
            home = (c.x, c.y)
            remember(c, home=home)
        mm = w.move_mask(sc.eid)
        opts = []
        for d in range(8):
            if mm[d]:
                dx, dy = DIR_VECTORS[d]
                if max(abs(c.x + dx - home[0]), abs(c.y + dy - home[1])) <= 6:
                    opts.append(d)
        if not opts or jitter(w, sc.eid, 7) < 0.4 or c.ts.moved_sq >= 2:
            return end_turn() if c.ts.moved_sq else None if not opts else end_turn()
        d = opts[int(jitter(w, sc.eid, 11) * len(opts)) % len(opts)]
        return Act(ActKind.MOVE, d)

    @staticmethod
    def _behind_door(sc: Scene, tgt) -> bool:
        """True if the target square is not reachable without opening a door."""
        c, w = sc.c, sc.w
        cost_ap, _ = tactics.cost_grids(sc)
        dist = geo.dijkstra(cost_ap, c.x, c.y, 1e6, 1 if w.rules.diagonal_rule == "5-10-5" else 0)
        x0, y0 = max(0, tgt[0] - 1), max(0, tgt[1] - 1)
        return not np.isfinite(dist[y0:tgt[1] + 2, x0:tgt[0] + 2]).any()

    def hunt(self, sc: Scene, prof: Profile, gate: str | None) -> Act | None:
        """In combat but blind: head for where allies last saw the enemy (shared alarm)."""
        c, w = sc.c, sc.w
        if gate in ("hold", "sleep") or not tactics.mobile(sc):
            return None
        best = None
        for al in sc.allies:
            for e, (x, y, r) in al.aware.items():
                if 0 <= e < len(w.entities) and w.entities[e].present and w.relation(sc.eid, e) == Relation.HOSTILE:
                    if best is None or r > best[2]:
                        best = (x, y, r)
        if best is None:
            return None
        if True:
            from emberdeep.ai.explore import door_toward
            dd = door_toward(sc, best[0], best[1]) if self._behind_door(sc, best) else None
            if dd is not None:
                return dd
        cost_ap, cost_wk = tactics.cost_grids(sc)
        rule = 1 if w.rules.diagonal_rule == "5-10-5" else 0
        dist = geo.dijkstra(cost_wk, c.x, c.y, float(c.ts.movement_left), rule)
        tx, ty = best[0], best[1]
        # nearest reachable cell to the target
        ys, xs = np.nonzero(dist <= c.ts.movement_left)
        if len(xs) <= 1:
            return None
        d = np.maximum(np.abs(xs - tx), np.abs(ys - ty))
        i = int(np.argmin(d))
        if (int(xs[i]), int(ys[i])) == (c.x, c.y):
            return None
        return tactics.step_toward(sc, dist, cost_wk, int(xs[i]), int(ys[i]))
