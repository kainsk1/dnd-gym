"""Morale, fleeing, surrender, truce offers / acceptance, parley and bribes for scripted creatures.

Engine hooks used: ``world.morale_check`` (WIS save; sets FLEEING or SURRENDERED according to ``can_surrender``),
``world.set_fleeing``, ``world.set_surrendered``, the universal abilities ``offer_truce`` / ``accept_truce`` /
``parley`` / ``surrender`` and the books ``world.bribes`` (gold given between factions) and ``Creature.gold_from``.

Rules of thumb implemented here (all documented in ``docs/AI.md``)
-------------------------------------------------------------------
morale     checked when hp fraction < ``creature.morale`` (or at the first wound for cowards) *and* hp has dropped since
           the last check, or when half the creature's faction has fallen.  Fearless (morale 0) creatures never check.
fleeing    disengage when in reach of a foe, otherwise run toward the nearest exit marker (APPROACH) or away from foes;
           interact with an adjacent exit to escape; rally (stop fleeing) after two quiet turns; surrender when cornered
           if the creature can.
surrender  a surrendered creature only ends its turn (the engine clears the flag if it is attacked).
truce      accepts an offer if (a) bribed at least ``bribe_price``, (b) the faction is losing (power < 70 % of the
           offerer's or average hp < 45 %), or (c) the offerer is at least as strong and never harmed us; refuses
           offers from factions that betrayed it.  Offers one when losing badly; parleys when outmatched.
"""
from __future__ import annotations

import numpy as np

from emberdeep.ai.abinfo import U
from emberdeep.ai.base import end_turn, mem, remember, turn_mem, turn_remember
from emberdeep.ai.scene import Scene
from emberdeep.ai.tactics import cost_grids, flee_cell, mobile, step_toward
from emberdeep.ai.utility import ab_act
from emberdeep.core import resolution as R
from emberdeep.core._geom_fallback import geo
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import INCAP_MASK, S
from emberdeep.core.types import ControllerKind, EntityKind, Relation


# ---------------------------------------------------------------------------------------------------- power / hp
def faction_members(w, fid: int) -> list:
    return [m for m in w.creatures if m.faction == fid and m.present and m.conscious and not m.status & (
        S.DEAD | S.ESCAPED | S.SURRENDERED)]


def faction_hp_frac(w, fid: int) -> float:
    ms = faction_members(w, fid)
    if not ms:
        return 0.0
    return sum(m.hp / max(1, m.max_hp) for m in ms) / len(ms)


def bribe_price(w, fid: int) -> float:
    """Gold that makes a faction willing to talk: 10 gp per member weighted by challenge (min 0.5)."""
    return 10.0 * sum(max(m.cr, float(m.level) * 0.5, 0.5) for m in faction_members(w, fid))


def is_losing(w, c) -> bool:
    mine = w.faction_power(c.faction)
    foe = 0.0
    for f in range(len(w.factions)):
        if f != c.faction and w.relation_f(c.faction, f) == Relation.HOSTILE:
            foe += w.faction_power(f)
    return mine < 0.7 * foe or (foe > 0 and faction_hp_frac(w, c.faction) < 0.45)


# ---------------------------------------------------------------------------------------------------- truce
def distrusts(w, c, fid: int) -> bool:
    """Has a member of faction ``fid`` betrayed us (attacked under truce)?"""
    for e in c.betrayed_by:
        if 0 <= e < len(w.entities) and w.entities[e].kind == EntityKind.CREATURE and w.entities[e].faction == fid:
            return True
    return False


def hostile_recently(w, c, fid: int) -> bool:
    for e in c.grudges:
        if 0 <= e < len(w.entities) and w.entities[e].kind == EntityKind.CREATURE and w.entities[e].faction == fid:
            return True
    return False


def should_accept(sc: Scene, prof, offerer: int) -> bool:
    w, c = sc.w, sc.c
    if distrusts(w, c, offerer):
        return False
    mine = w.faction_power(c.faction)
    theirs = w.faction_power(offerer)
    bribe = w.bribes.get((offerer, c.faction), 0)
    if bribe >= bribe_price(w, c.faction):
        return True
    fearless = c.morale <= 0 and not c.is_pc
    if fearless:
        return mine < 0.4 * theirs
    if mine < 0.7 * theirs or faction_hp_frac(w, c.faction) < 0.45:
        return True
    if theirs >= 0.9 * mine and not hostile_recently(w, c, offerer) and not c.ever_hostile:
        return True
    return False


def _first_of_faction(w, fid: int):
    for m in w.creatures:
        if m.faction == fid and m.present and not m.status & S.DEAD:
            return m
    return None


def truce_response(sc: Scene, prof) -> Act | None:
    """Accept a pending truce offer addressed to our faction (free action, once per turn)."""
    w, c = sc.w, sc.c
    if not w.truce_offers:
        return None
    tm = turn_mem(w, c)
    if tm.get("truce_done"):
        return None
    k = U["accept_truce"]
    mask = sc.mask()
    if not mask[k]:
        return None
    for (fa, fb, r) in w.truce_offers:
        if fb == c.faction and w.has_truce_offer(fa, fb):
            o = _first_of_faction(w, fa)
            if o is None:
                continue
            turn_remember(w, c, truce_done=True)
            if should_accept(sc, prof, fa):
                return ab_act(k, o.eid)
    return None


def truce_initiative(sc: Scene, prof) -> Act | None:
    """Offer a truce / parley when losing badly (or bribed) -- only intelligent creatures talk."""
    w, c = sc.w, sc.c
    if c.scores[3] < 4 or not sc.vis:
        return None
    m = mem(c)
    last = m.get("offered", -99)
    if w.round - last < 4 or m.get("tr_chk") == (w.round, w.mode):
        return None
    remember(c, tr_chk=(w.round, w.mode))
    mine = w.faction_power(c.faction)
    foe_power = {}
    for f in sc.standing:
        foe_power[f.faction] = w.faction_power(f.faction)
    if not foe_power:
        return None
    tf = max(foe_power, key=foe_power.get)
    theirs = foe_power[tf]
    bribed = w.bribes.get((tf, c.faction), 0) >= bribe_price(w, c.faction) > 0
    desperate = mine < 0.5 * theirs and faction_hp_frac(w, c.faction) < 0.6
    if not (bribed or desperate):
        return None
    if w.has_truce_offer(c.faction, tf):
        return None
    mask = sc.mask()
    abil = w.abilities(sc.eid)
    tgt = None
    for f in sc.vis:
        if f.faction == tf:
            tgt = f
            break
    if tgt is None:
        return None
    k = U["offer_truce"]
    if mask[k] and abil[k].valid_target(w, c, tgt.eid):
        remember(c, offered=w.round)
        return ab_act(k, tgt.eid)
    k = U["parley"]
    if desperate and mask[k] and abil[k].valid_target(w, c, tgt.eid):
        remember(c, offered=w.round)
        return ab_act(k, tgt.eid)
    return None


# ---------------------------------------------------------------------------------------------------- morale
def check_morale(sc: Scene, prof) -> None:
    """Possibly break morale (sets FLEEING / SURRENDERED through the engine API)."""
    w, c = sc.w, sc.c
    if not w.rules.morale or c.status & (S.FLEEING | S.SURRENDERED) or c.morale <= 0 or c.is_pc and not c.morale:
        return
    if c.controller != ControllerKind.SCRIPTED:
        return
    m = mem(c)
    hpf = c.hp / max(1, c.max_hp)
    last_hp = m.get("morale_hp", c.max_hp)
    last_dead = m.get("morale_dead", 0)
    trigger = False
    thr = c.morale if not prof.flee_wound else 0.999
    if hpf < thr and c.hp < last_hp:
        trigger = True
    members = [x for x in w.creatures if x.faction == c.faction and x is not c]
    dead = sum(1 for x in members if x.status & (S.DEAD | S.DYING) or not x.present)
    if members and dead * 2 >= len(members) + 1 and dead > last_dead and hpf < 0.9:
        trigger = True
    remember(c, morale_hp=c.hp, morale_dead=dead)
    if trigger:
        w.morale_check(c)


# ---------------------------------------------------------------------------------------------------- fleeing
def exit_entities(w) -> list:
    return [o for o in w.objects if o.kind == EntityKind.EXIT and o.present]


def flee_act(sc: Scene, prof) -> Act | None:
    """One action for a creature with FLEEING set.  ``None`` means: cornered, fight normally."""
    w, c = sc.w, sc.c
    # rally after two quiet turns
    quiet = mem(c).get("quiet", 0)
    near = [f for f in sc.standing if w.distance(sc.eid, f.eid) <= 12 and (f in sc.vis or f.eid in c.aware)]
    if not near and w.round > 1:
        quiet += 1
        remember(c, quiet=quiet)
        if quiet >= 2:
            w.set_fleeing(c, False)
            remember(c, quiet=0, morale_hp=c.hp)
            return None
    else:
        remember(c, quiet=0)
    abil = w.abilities(sc.eid)
    mask = sc.mask()
    # leave through an adjacent exit
    k = U["interact"]
    exits = exit_entities(w)
    if mask[k]:
        for e in exits:
            if w.distance(sc.eid, e.eid) <= 1 and abil[k].valid_target(w, c, e.eid):
                return ab_act(k, e.eid)
    adj = [f for f in sc.standing if w.distance(sc.eid, f.eid) <= max(1, sc.offense(f)["reach"])]
    can_run = mobile(sc) and w.move_mask(sc.eid).any()
    if not can_run and not (mask[U["dash"]] and not c.ts.action_used and c.ts.movement_left == 0):
        if adj:
            if c.can_surrender and mask[U["surrender"]]:
                return ab_act(U["surrender"])
            return None
        return end_turn()
    if adj and mask[U["disengage"]] and not c.ts.disengaged:
        return ab_act(U["disengage"])
    # run for the exit
    if exits and can_run:
        cost_ap, cost_wk = cost_grids(sc)
        rule = 1 if w.rules.diagonal_rule == "5-10-5" else 0
        dist = geo.dijkstra(cost_ap, c.x, c.y, 1e6, rule)
        best = None
        for e in exits:
            d = float(dist[max(0, e.y - 1):e.y + 2, max(0, e.x - 1):e.x + 2].min())
            if best is None or d < best[0]:
                best = (d, e)
        if best is not None and best[0] < float("inf"):
            if w.distance(sc.eid, best[1].eid) > 1 and w.approach_mask(sc.eid, [best[1].eid])[0]:
                return Act(ActKind.APPROACH, -1, best[1].eid)
    if can_run:
        cost_ap, cost_wk = cost_grids(sc)
        rule = 1 if w.rules.diagonal_rule == "5-10-5" else 0
        dist_wk = geo.dijkstra(cost_wk, c.x, c.y, float(c.ts.movement_left), rule)
        cell = flee_cell(sc, dist_wk, None)
        if cell is not None and cell != (c.x, c.y):
            step = step_toward(sc, dist_wk, cost_wk, cell[0], cell[1])
            if step is not None:
                return step
    if not c.ts.action_used and mask[U["dash"]] and near:
        return ab_act(U["dash"])
    if adj:
        if c.can_surrender and mask[U["surrender"]]:
            return ab_act(U["surrender"])
        return None
    return end_turn()


# ---------------------------------------------------------------------------------------------------- entry point
def pre_turn(sc: Scene, prof) -> Act | None:
    """Social / morale layer run at the start of every act.  Returns an act to take, or None to continue."""
    w, c = sc.w, sc.c
    a = truce_response(sc, prof)
    if a is not None:
        return a
    if prof.style == "vip":                 # escort NPCs keep walking; they never break morale
        return None
    if c.status & S.SURRENDERED:
        return end_turn()
    if not (c.status & S.FLEEING):
        check_morale(sc, prof)
        if c.status & S.SURRENDERED:
            return end_turn()
    if c.status & S.FLEEING:
        return flee_act(sc, prof) or None
    if sc.standing and sc.vis:
        a = truce_initiative(sc, prof)
        if a is not None:
            return a
    return None
