"""Objective evaluation and episode outcome (DESIGN §4.15)."""
from __future__ import annotations

from emberdeep.core.entities import S

from emberdeep.core.events import EventType as ET
from emberdeep.core.types import ControllerKind, EntityKind, ObjectiveKind, Relation, Status


def add_objective(world, spec) -> int:
    """Register an ``ObjectiveSpec``; returns its index."""
    from emberdeep.core.world import ObjectiveState
    facs = []
    for f in spec.factions:
        try:
            facs.append(world.faction_index(f))
        except KeyError:
            pass
    st = ObjectiveState(spec, len(world.objectives), facs)
    world.objectives.append(st)
    return st.index


def _players(world, st) -> list[int]:
    if st.faction_idx:
        return list(st.faction_idx)
    out = []
    for c in world.creatures:
        if c.controller != ControllerKind.SCRIPTED and c.faction not in out:
            out.append(c.faction)
    return out


def _members(world, facs):
    return [c for c in world.creatures if c.faction in facs]


def _gone(c) -> bool:
    return (not c.present and not c.status & S.DYING) or bool(c.status & (S.DEAD | S.ESCAPED |
                                                                              S.SURRENDERED | S.FLEEING))


def _match(world, c, targets) -> bool:
    for t in targets:
        if t == c.template_id or t in c.tags or t == c.name or t == c.agent_id:
            return True
    return False


def _target_factions(world, st, players):
    out = []
    for t in st.spec.targets:
        try:
            out.append(world.faction_index(t))
        except KeyError:
            pass
    if not out:
        for f in range(len(world.factions)):
            if f not in players and any(world.relation_f(f, p) == Relation.HOSTILE for p in players):
                out.append(f)
    return out


def _check(world, st):
    """Returns (done, failed, completer_eid)."""
    spec = st.spec
    players = _players(world, st)
    members = _members(world, players)
    k = spec.kind
    if k == ObjectiveKind.DEFEAT_ALL:
        tf = _target_factions(world, st, players)
        foes = [c for c in world.creatures if c.faction in tf]
        if foes and all(_gone(c) for c in foes):
            alive = [m for m in members if m.present and not m.status & S.DEAD]
            return True, False, alive[0].eid if alive else -1
        return False, False, -1
    if k == ObjectiveKind.DEFEAT_TARGET:
        tg = [c for c in world.creatures if _match(world, c, spec.targets)]
        if tg and all(c.status & S.DEAD for c in tg):
            killer = tg[0].killed_by
            return True, False, killer
        return False, False, -1
    if k == ObjectiveKind.REACH_ZONE:
        cells = set()
        for r in spec.targets:
            cells.update(tuple(p) for p in world.regions.get(r, ()))
        live = [m for m in members if m.present and not m.status & S.DEAD]
        inz = [m for m in live if any(p in cells for p in m.cells())]
        if spec.required_all:
            ok = bool(live) and len(inz) == len(live)
        else:
            ok = bool(inz)
        return ok, False, inz[0].eid if inz else -1
    if k == ObjectiveKind.COLLECT:
        for m in members:
            if m.status & S.DEAD:
                continue
            if any(i.id in spec.targets for i in m.inventory):
                return True, False, m.eid
        return False, False, -1
    if k == ObjectiveKind.ESCAPE:
        esc = [m for m in members if m.status & S.ESCAPED]
        if spec.required_all:
            rest = [m for m in members if not m.status & (S.ESCAPED | S.DEAD)]
            ok = bool(esc) and not rest
        else:
            ok = bool(esc)
        return ok, False, esc[0].eid if esc else -1
    if k == ObjectiveKind.SURVIVE:
        alive = [m for m in members if not m.status & S.DEAD]
        if not alive:
            return False, True, -1
        return world.round >= spec.value, False, alive[0].eid
    if k == ObjectiveKind.PROTECT:
        tg = [c for c in world.creatures if _match(world, c, spec.targets)]
        if any(c.status & S.DEAD for c in tg):
            return False, True, -1
        if spec.value and world.round >= spec.value:
            return True, False, tg[0].eid if tg else -1
        return False, False, -1
    if k == ObjectiveKind.LOOT_VALUE:
        tot = 0.0
        best = None
        for m in members:
            if m.status & S.DEAD:
                continue
            tot += m.wealth
            best = m
        return tot >= spec.value, False, best.eid if best else -1
    return False, False, -1


def evaluate(world) -> None:
    """Evaluate all active objectives after an atomic action; emits OBJECTIVE and may end the episode."""
    if not world.objectives or world.done:
        return
    for st in world.objectives:
        if st.done or st.failed:
            continue
        done, failed, who = _check(world, st)
        if not (done or failed):
            continue
        st.done, st.failed, st.completer = done, failed, who
        world.emit(ET.OBJECTIVE, who, -1, 0, st.index,
                   {"name": st.spec.name, "team": _players(world, st), "success": done})
        if st.spec.ends_episode:
            world._finish("objective" if done else "objective_failed")
            return


def winners(world) -> list[int]:
    """Faction indices that won: owners of completed episode-ending objectives; else the sole survivor faction."""
    w: list[int] = []
    for st in world.objectives:
        if st.done and st.spec.ends_episode:
            for f in _players(world, st):
                if f not in w:
                    w.append(f)
    if w:
        return w
    if world.objectives and any(o.failed for o in world.objectives):
        return []
    alive = {c.faction for c in world.creatures if c.present and not c.status & (S.DEAD | S.SURRENDERED)}
    return sorted(alive) if len(alive) == 1 else []
