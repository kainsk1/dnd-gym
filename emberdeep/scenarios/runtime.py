"""Scenario runtime: waves, triggers, sleepers, scenario-specific payoffs and metrics.

``ScenarioRuntime`` is stored on the world as ``world.scenario_runtime``.  It is **immutable configuration**
(the spec plus a few lookup tables): ``World.clone`` shares it by reference, and every piece of mutable
runtime state lives in flat, scalar-valued ``world.stats`` entries prefixed ``"scn:"`` (``World.clone`` and
``restore`` copy ``stats`` shallowly, so clones evolve independently).  Nothing here holds a lambda.

The env calls :meth:`ScenarioRuntime.after_step` after **every** ``world.step`` (and once after the build, which
``build_world`` already does).  It is idempotent and catches up on missed rounds, so waves fire exactly at the
start of their target round: when the step that crosses the round boundary returns, the first creature of the new
round has not acted yet.  Call it *before* draining events (``World.drain_events``): event-based metrics read the
undrained tail of ``world.events``.

``spec.runtime`` keys (all optional, plain data)::

    "wake_radius": 3                        sleepers within this many cells of a hostile may wake each round
    "track_regions": ["bridge"]             count agent-rounds spent in these regions (metrics)
    "escort": {"tag": "vip", "fee": 150, "faction": "party", "region": "exit"}
                                            fee (gp) for each surviving agent if the tagged creature escapes/arrives
    "last_standing": {"objective": "last_standing", "prize": 1000}
                                            end the episode when every surviving agent faction is mutually
                                            non-hostile and no scripted hostile remains; prize split among survivors
    "relic_escape": {"item": "ember_relic"} the faction whose member escapes carrying the item wins
                                            (completes the placeholder objective ``"relic_<faction id>"``)
    "survival_bonus": 0.02                  terminal reward per round an agent stayed alive
    "bribe": {"target": "captain", "threshold": 100, "payers": "party", "foes": "bandits"}
                                            gold given to creatures tagged ``target`` by ``payers`` members reaching
                                            ``threshold`` gp turns the ``payers``/``foes`` relation into TRUCE
    "track_loot_regions": ["alcove_north"]  per-agent gold picked up from objects inside these regions (metrics)
    "populate": "module:function"           procedural population hook, see builder.build_world
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Condition, ControllerKind, GameMode, Relation
from emberdeep.scenarios.spec import ScenarioSpec, WaveSpec

K = "scn:"          # prefix of runtime state keys in ``world.stats``


# =============================================================================== sleepers
def mark_asleep(world, c) -> None:
    """Put ``c`` to sleep: UNCONSCIOUS (ends on damage); the runtime wakes it on noise/proximity/triggers."""
    R.add_condition(world, c, Condition.UNCONSCIOUS, -1, -1, None, 10, ends_on_damage=True, silent=True)
    cur = tuple(world.stats.get(K + "asleep", ()))
    if c.eid not in cur:
        world.stats[K + "asleep"] = cur + (c.eid,)


def wake_creature(world, c, reason: str = "") -> bool:
    """Wake a sleeper (removes the sleep condition and the prone it implied).  Returns True if it was asleep."""
    cur = tuple(world.stats.get(K + "asleep", ()))
    if c.eid not in cur:
        return False
    world.stats[K + "asleep"] = tuple(e for e in cur if e != c.eid)
    if "asleep" in c.tags:
        c.tags = tuple(t for t in c.tags if t != "asleep")      # the scripted AI gates on the tag
    if c.cond_mask >> int(Condition.UNCONSCIOUS) & 1 and not c.status & (S.DYING | S.STABLE | S.DEAD):
        R.remove_condition(world, c, Condition.UNCONSCIOUS, silent=True)
        R.remove_condition(world, c, Condition.PRONE, silent=True)
    world.emit(ET.REVEALED, -1, c.eid, 0, -1, {"reason": "wake:" + reason})
    world.log(f"{c.name} wakes up!", "warn", c.eid)
    world.vis_dirty()
    return True


def is_asleep(world, c) -> bool:
    """True while the runtime considers ``c`` asleep."""
    return c.eid in tuple(world.stats.get(K + "asleep", ()))


def _gini(values: list[float]) -> float:
    vals = sorted(max(0.0, float(v)) for v in values)
    n = len(vals)
    tot = sum(vals)
    if n == 0 or tot <= 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return float((2.0 * cum) / (n * tot) - (n + 1.0) / n)


def join_turn_order(world, c) -> None:
    """Insert a newly spawned creature into the running turn order (initiative-sorted in combat)."""
    if c.eid in world.order:
        return
    if world.mode == int(GameMode.COMBAT):
        c.initiative = D.d20(world.rng) + c.mods[1]
        pos = len(world.order)
        for j, e in enumerate(world.order):
            if world.entities[e].initiative < c.initiative:
                pos = j
                break
        world.order.insert(pos, c.eid)
        if pos <= world.turn_idx:
            world.turn_idx += 1
    else:
        world.order.append(c.eid)


class ScenarioRuntime:
    """Static scenario logic bound to a world (see module docstring)."""

    def __init__(self, spec: ScenarioSpec, agents: dict[str, int], faction_index: dict[str, int],
                 player_factions: list[int]):
        self.spec = spec
        self.cfg: dict[str, Any] = dict(spec.runtime)
        self.waves: list[WaveSpec] = list(spec.waves)
        self.agents: dict[str, int] = dict(agents)
        self.agent_of: dict[int, str] = {e: a for a, e in agents.items()}
        self.faction_index: dict[str, int] = dict(faction_index)
        self.player_factions: tuple[int, ...] = tuple(player_factions)

    # ------------------------------------------------------------------ world hooks
    def after_step(self, world) -> None:
        """Idempotent post-step upkeep: events, sleepers, round-start waves, triggers, special endings."""
        if not world.started:
            return
        st = world.stats
        spawned = False
        self._scan_events(world)
        done_r = int(st.get(K + "round_done", 0))
        if not world.done:
            for r in range(done_r + 1, int(world.round) + 1):
                spawned |= self._round_start(world, r)
                st[K + "round_done"] = r
            self._wake_sleepers(world)
            spawned |= self._trigger_waves(world)
        self._tick_altars(world)
        self._special(world)
        if spawned and not world.done:
            world._settle()

    # ------------------------------------------------------------------ events
    def _scan_events(self, world) -> None:
        st = world.stats
        seen = int(st.get(K + "ev_seen", 0))
        total = int(world._ev_total)
        start = max(seen, int(world._ev_drained)) - int(world._ev_drained)
        if start < len(world.events):
            pf = self.player_factions
            for ev in world.events[start:]:
                t = ev.t
                if t == ET.TRUCE_ACCEPT:
                    st[K + "truces"] = st.get(K + "truces", 0) + 1
                    st.setdefault(K + "first_truce_round", ev.round)
                elif t == ET.BETRAYAL:
                    st[K + "betrayals"] = st.get(K + "betrayals", 0) + 1
                    st.setdefault(K + "first_betrayal_round", ev.round)
                elif t == ET.DAMAGE and ev.src >= 0 and ev.dst >= 0:
                    a, b = world.entities[ev.src], world.entities[ev.dst]
                    if getattr(a, "faction", -1) in pf and getattr(b, "faction", -1) in pf \
                            and a.faction != b.faction:
                        st[K + "pvp_damage"] = st.get(K + "pvp_damage", 0) + ev.amount
                    if "boss" in getattr(b, "tags", ()) and ev.src in self.agent_of:
                        key = f"{K}boss_dmg:{ev.src}"
                        st[key] = st.get(key, 0) + ev.amount
                elif t == ET.PICKUP and ev.src in self.agent_of and ev.dst >= 0 and ev.amount > 0:
                    o = world.entities[ev.dst]
                    for reg in self.cfg.get("track_loot_regions", ()):
                        if (o.x, o.y) in {tuple(p) for p in world.regions.get(reg, ())}:
                            key = f"{K}loot:{reg}:{ev.src}"
                            st[key] = st.get(key, 0) + ev.amount
        st[K + "ev_seen"] = total

    # ------------------------------------------------------------------ rounds / waves
    def _round_start(self, world, r: int) -> bool:
        spawned = False
        for i, w in enumerate(self.waves):
            if w.at_round <= 0 or w.trigger_region or w.trigger_item:
                continue
            total = max(1, int(w.repeat)) if w.every > 0 else 1
            if r < w.at_round:
                continue
            due = 1 if w.every <= 0 else min(total, (r - w.at_round) // w.every + 1)
            fired = int(world.stats.get(f"{K}w{i}", 0))
            while fired < due:
                spawned |= bool(self._fire(world, i, w, fired))
                fired += 1
                world.stats[f"{K}w{i}"] = fired
        for reg in self.cfg.get("track_regions", ()):
            cells = {tuple(c) for c in world.regions.get(reg, ())}
            for aid, eid in self.agents.items():
                c = world.entities[eid]
                if c.present and not c.status & (S.DEAD | S.ESCAPED) and any(p in cells for p in c.cells()):
                    key = f"{K}occ:{reg}:{eid}"
                    world.stats[key] = world.stats.get(key, 0) + 1
        self._wake_by_proximity(world)
        return spawned

    def _player_side(self, world, c) -> bool:
        return c.controller != ControllerKind.SCRIPTED or c.faction in self.player_factions

    def _trigger_waves(self, world) -> bool:
        spawned = False
        for i, w in enumerate(self.waves):
            if not (w.trigger_region or w.trigger_item) or world.stats.get(f"{K}w{i}", 0):
                continue
            hit = False
            if w.trigger_region:
                cells = {tuple(c) for c in world.regions.get(w.trigger_region, ())}
                for c in world.creatures:
                    if c.present and not c.status & (S.DEAD | S.ESCAPED) and self._player_side(world, c) \
                            and any(p in cells for p in c.cells()):
                        hit = True
                        break
            if not hit and w.trigger_item:
                for c in world.creatures:
                    if c.present and not c.status & S.DEAD and c.find_item(w.trigger_item) is not None:
                        hit = True
                        break
            if hit:
                world.stats[f"{K}w{i}"] = 1
                spawned |= bool(self._fire(world, i, w, 0))
        return spawned

    def _fire(self, world, idx: int, wave: WaveSpec, k: int) -> list[int]:
        from emberdeep.scenarios.builder import spawn_spec
        eids: list[int] = []
        extra = int(math.floor(wave.scale_with_round * k + 0.5))
        for sp in wave.spawns:
            sp2 = dataclasses.replace(sp, count=max(0, sp.count + extra))
            eids += spawn_spec(world, self.spec, sp2, world.rng)
        for e in eids:
            c = world.entities[e]
            self._alert(world, c)
            join_turn_order(world, c)
            world.emit(ET.SUMMON, -1, e, 0, -1, {"wave": idx, "reason": "wave", "round": int(world.round)})
        if wave.announce:
            world.log(wave.announce, "warn")
        world.stats[K + "wave_spawns"] = world.stats.get(K + "wave_spawns", 0) + len(eids)
        world.stats[K + "waves_fired"] = world.stats.get(K + "waves_fired", 0) + 1
        return eids

    @staticmethod
    def _alert(world, c) -> None:
        """Wave arrivals come hunting: they know where every hostile creature stands (unless told to hold/sleep) and
        the arrival starts combat."""
        if any(t in ("stationary", "asleep") or t.startswith("guard:") for t in c.tags):
            return
        hostile = False
        for o in world.creatures:
            if o.present and o is not c and not o.status & S.DEAD and world.relation(c.eid, o.eid) == Relation.HOSTILE:
                c.aware[o.eid] = (o.x, o.y, int(world.round))
                o.aware[c.eid] = (c.x, c.y, int(world.round))      # the alarm is raised: nobody is surprised
                hostile = True
        if hostile and world.mode == int(GameMode.EXPLORATION):
            world._combat_pending = True                           # scripted hunters only move in combat mode
            world._combat_trigger = c.eid

    # ------------------------------------------------------------------ sleepers
    def _wake_sleepers(self, world) -> None:
        asleep = tuple(world.stats.get(K + "asleep", ()))
        if not asleep:
            return
        hearing = int(world.rules.hearing_radius)
        in_combat = world.mode == int(GameMode.COMBAT)
        for eid in asleep:
            c = world.entities[eid]
            if not c.present or c.status & S.DEAD:
                world.stats[K + "asleep"] = tuple(e for e in world.stats[K + "asleep"] if e != eid)
                continue
            if not c.cond_mask >> int(Condition.UNCONSCIOUS) & 1:      # damaged: the engine removed the condition
                wake_creature(world, c, "damage")
                continue
            reason = ""
            for tag in c.tags:
                if tag.startswith("wake:"):
                    cells = {tuple(p) for p in world.regions.get(tag[5:], ())}
                    for o in world.creatures:
                        if o.present and not o.status & (S.DEAD | S.ESCAPED) and self._player_side(world, o) \
                                and o.eid != eid and any(p in cells for p in o.cells()):
                            reason = "intruder"
                            break
                if reason:
                    break
            if not reason and in_combat:
                for o in world.creatures:
                    if o.present and o.eid != eid and o.eid not in asleep and o.conscious \
                            and world.relation(eid, o.eid) == Relation.HOSTILE and o.aware \
                            and max(abs(o.x - c.x), abs(o.y - c.y)) <= hearing:
                        reason = "combat_noise"
                        break
            if reason:
                wake_creature(world, c, reason)

    def _wake_by_proximity(self, world) -> None:
        """Once per round: sleepers near a hostile may notice it (certain when adjacent)."""
        asleep = tuple(world.stats.get(K + "asleep", ()))
        if not asleep:
            return
        radius = int(self.cfg.get("wake_radius", 3))
        for eid in asleep:
            c = world.entities[eid]
            if not c.present or c.status & S.DEAD:
                continue
            best = 99
            for o in world.creatures:
                if o.present and o.eid != eid and o.conscious and eid != o.eid \
                        and world.relation(eid, o.eid) == Relation.HOSTILE:
                    best = min(best, max(abs(o.x - c.x), abs(o.y - c.y)))
            if best <= 1 or (best <= radius and world.rng.random() < 0.5):
                wake_creature(world, c, "noise")

    # ------------------------------------------------------------------ altars (idempotent upkeep)
    def _tick_altars(self, world) -> None:
        from emberdeep.core.types import EntityKind
        from emberdeep.rules.registry import OBJECT_HOOKS
        for o in world.objects:
            if o.kind == EntityKind.ALTAR and o.present and o.hook.startswith("scn_"):
                h = OBJECT_HOOKS.get(o.hook)
                if h is not None:
                    h().tick(world, o)

    # ------------------------------------------------------------------ special endings
    def _special(self, world) -> None:
        if world.done:
            return
        esc = self.cfg.get("escort")
        if esc and not world.stats.get(K + "vip_round"):
            cells = {tuple(c) for c in world.regions.get(esc.get("region", ""), ())} if esc.get("region") else set()
            for c in world.creatures:
                if esc["tag"] in c.tags and not c.status & S.DEAD and (
                        c.status & S.ESCAPED or (cells and c.present and any(p in cells for p in c.cells()))):
                    world.stats[K + "vip_round"] = int(world.round)
                    world.log(f"{c.name} reaches safety!", "info", c.eid)
                    for ob in world.objectives:
                        if ob.spec.name == esc.get("objective", "vip_safe") and not ob.done and not ob.failed:
                            facs = [world.faction_index(f) for f in ob.spec.factions] or list(self.player_factions)
                            self._complete(world, ob, facs, c.eid, "vip_safe")
                            return
                    break
        br = self.cfg.get("bribe")
        if br and not world.stats.get(K + "bribe_round"):
            payers = world.faction_index(br["payers"])
            foes = world.faction_index(br["foes"])
            tot = self._bribe_total(world, br)
            if tot >= int(br["threshold"]) and world.relation_f(payers, foes) == Relation.HOSTILE:
                world.stats[K + "bribe_round"] = int(world.round)
                world.set_relation(payers, foes, Relation.TRUCE)
                for o in world.creatures:
                    o.aware = {k: v for k, v in o.aware.items() if world.relation(o.eid, k) == Relation.HOSTILE}
                world.log("The captain pockets the toll and calls off his men.", "info")
        re_ = self.cfg.get("relic_escape")
        if re_:
            for c in world.creatures:
                if c.status & S.ESCAPED and not c.status & S.DEAD and c.find_item(re_["item"]) is not None:
                    fid = world.factions[c.faction][0]
                    for st in world.objectives:
                        if st.spec.name == f"relic_{fid}" and not st.done:
                            self._complete(world, st, [c.faction], c.eid, "relic_escape")
                            return
        ls = self.cfg.get("last_standing")
        if ls and world.round >= int(ls.get("min_round", 1)):
            alive = [world.entities[e] for e in self.agents.values()]
            alive = [c for c in alive if c.present and not c.status & (S.DEAD | S.ESCAPED | S.SURRENDERED)]
            if not alive:
                return
            facs = sorted({c.faction for c in alive})
            if any(c.present and not c.status & (S.DEAD | S.SURRENDERED | S.FLEEING | S.ESCAPED)
                   and c.controller == ControllerKind.SCRIPTED and c.faction not in facs
                   and any(world.relation_f(c.faction, f) == Relation.HOSTILE for f in facs)
                   for c in world.creatures):
                return
            if all(world.relation_f(a, b) in (Relation.TRUCE, Relation.ALLIED)
                   for i, a in enumerate(facs) for b in facs[i + 1:]):
                for st in world.objectives:
                    if st.spec.name == ls.get("objective", "last_standing") and not st.done:
                        self._complete(world, st, facs, alive[0].eid, "last_standing")
                        return

    def _bribe_total(self, world, br) -> int:
        payers = world.faction_index(br["payers"])
        pay_eids = {c.eid for c in world.creatures if c.faction == payers}
        tot = 0
        for c in world.creatures:
            if br["target"] in c.tags:
                tot += sum(v for e, v in c.gold_from.items() if e in pay_eids)
        return int(tot)

    @staticmethod
    def _complete(world, st, factions: list[int], who: int, reason: str) -> None:
        st.faction_idx = list(factions)
        st.done = True
        st.completer = who
        world.emit(ET.OBJECTIVE, who, -1, 0, st.index,
                   {"name": st.spec.name, "team": list(factions), "success": True})
        if st.spec.ends_episode:
            world._finish(reason)

    # ------------------------------------------------------------------ payoffs
    def terminal_payoffs(self, world) -> dict[str, dict[str, float]]:
        """Scenario terminal payoffs per agent id, in **reward units** (gp x ``rewards.terminal_scale``).

        Components: ``escort_fee``, ``prize`` and ``survival`` (the generic ``RewardSpec.terminal_payoff`` gold payoff is
        computed by the env's reward engine, not here).  Values use ``rewards.terminal_scale`` of the *spec*; the env
        pays components it does not know at weight 1.  Meaningful once the episode is over; before that it reports
        what the agent would receive if the episode ended now."""
        rw = self.spec.rewards
        scale = float(rw.terminal_scale)
        out: dict[str, dict[str, float]] = {}
        esc = self.cfg.get("escort")
        vip_ok = bool(esc) and bool(world.stats.get(K + "vip_round"))
        if esc:
            vip_ok = vip_ok and any(esc["tag"] in c.tags and not c.status & S.DEAD for c in world.creatures)
        winners_f: list[int] = []
        ls = self.cfg.get("last_standing")
        if ls and ls.get("prize"):
            for st in world.objectives:
                if st.spec.name == ls.get("objective", "last_standing") and st.done:
                    winners_f = list(st.faction_idx)
        survivors = [a for a, e in self.agents.items()
                     if world.entities[e].faction in winners_f and not world.entities[e].status & S.DEAD
                     and world.entities[e].present]
        for aid, eid in self.agents.items():
            c = world.entities[eid]
            dead = bool(c.status & S.DEAD)
            comp: dict[str, float] = {}
            if esc:
                fid = esc.get("faction")
                ok = (fid is None or c.faction == self.faction_index.get(fid, -1)) and not dead
                comp["escort_fee"] = float(esc.get("fee", 0)) * scale if (vip_ok and ok) else 0.0
            if ls and ls.get("prize"):
                comp["prize"] = float(ls["prize"]) * scale / len(survivors) if aid in survivors else 0.0
            bonus = float(self.cfg.get("survival_bonus", 0.0))
            if bonus:
                rounds = int(c.death_round) if dead and c.death_round >= 0 else int(world.round)
                comp["survival"] = bonus * rounds
            out[aid] = comp
        return out

    # ------------------------------------------------------------------ metrics
    def metrics(self, world) -> dict[str, float]:
        """Scenario-specific scalar metrics (see docs/SCENARIOS.md)."""
        from emberdeep.core.types import EntityKind
        st = world.stats
        m: dict[str, float] = {}
        ag = [world.entities[e] for e in self.agents.values()]
        m["rounds"] = float(world.round)
        m["agents_alive"] = float(sum(1 for c in ag if not c.status & S.DEAD and not c.status & S.ESCAPED))
        m["agents_dead"] = float(sum(1 for c in ag if c.status & S.DEAD))
        m["agents_escaped"] = float(sum(1 for c in ag if c.status & S.ESCAPED and not c.status & S.DEAD))
        m["wealth_gini"] = _gini([c.wealth for c in ag])
        m["gold_total"] = float(sum(c.wealth for c in ag))
        m["waves_fired"] = float(st.get(K + "waves_fired", 0))
        m["wave_spawns"] = float(st.get(K + "wave_spawns", 0))
        m["truces"] = float(st.get(K + "truces", 0))
        m["betrayals"] = float(st.get(K + "betrayals", 0))
        m["first_betrayal_round"] = float(st.get(K + "first_betrayal_round", -1))
        m["first_truce_round"] = float(st.get(K + "first_truce_round", -1))
        m["pvp_damage"] = float(st.get(K + "pvp_damage", 0))
        m["asleep_now"] = float(len(st.get(K + "asleep", ())))
        for aid in self.agents:
            c = world.entities[self.agents[aid]]
            m[f"alive/{aid}"] = 0.0 if c.status & S.DEAD else 1.0
        for reg in self.cfg.get("track_regions", ()):
            occ = [float(st.get(f"{K}occ:{reg}:{e}", 0)) for e in self.agents.values()]
            for aid, e in self.agents.items():
                m[f"occupancy/{reg}/{aid}"] = float(st.get(f"{K}occ:{reg}:{e}", 0))
            m[f"occupancy_max/{reg}"] = max(occ) if occ else 0.0
        bosses = [c for c in world.creatures if "boss" in c.tags]
        if bosses:
            m["bosses_total"] = float(len(bosses))
            m["bosses_dead"] = float(sum(1 for c in bosses if c.status & S.DEAD))
            dm = [float(st.get(f"{K}boss_dmg:{e}", 0)) for e in self.agents.values()]
            m["boss_engagers"] = float(sum(1 for d in dm if d > 0))
            m["boss_all_engaged"] = 1.0 if dm and all(d > 0 for d in dm) else 0.0
            for aid, e in self.agents.items():
                m[f"boss_damage/{aid}"] = float(st.get(f"{K}boss_dmg:{e}", 0))
        for reg in self.cfg.get("track_loot_regions", ()):
            vals = [float(st.get(f"{K}loot:{reg}:{e}", 0)) for e in self.agents.values()]
            m[f"loot_agents/{reg}"] = float(sum(1 for v in vals if v > 0))
            m[f"loot_gold/{reg}"] = float(sum(vals))
            for aid, e in self.agents.items():
                m[f"loot/{reg}/{aid}"] = float(st.get(f"{K}loot:{reg}:{e}", 0))
        br = self.cfg.get("bribe")
        if br:
            m["bribe/total"] = float(self._bribe_total(world, br))
            m["bribe/paid"] = 1.0 if st.get(K + "bribe_round") else 0.0
            for aid, e in self.agents.items():
                m[f"bribe/{aid}"] = float(sum(c.gold_from.get(e, 0) for c in world.creatures if br["target"] in c.tags))
        esc = self.cfg.get("escort")
        if esc:
            m["vip_arrived"] = 1.0 if st.get(K + "vip_round") else 0.0
            m["vip_arrival_round"] = float(st.get(K + "vip_round", -1))
            m["vip_alive"] = 1.0 if any(esc["tag"] in c.tags and not c.status & S.DEAD for c in world.creatures) else 0.0
        for o in world.objects:
            if o.kind != EntityKind.ALTAR or not o.hook.startswith("scn_"):
                continue
            s = o.state
            if o.hook == "scn_fountain":
                uses = [float(s.get(f"use:{e}", 0)) for e in self.agents.values()]
                m["fountain/charges"] = float(o.charges)
                m["fountain/level"] = float(s.get("c", 0.0))
                m["fountain/mean_level"] = float(s.get("csum", 0.0)) / max(1, int(s.get("cn", 0)))
                m["fountain/uses"] = float(s.get("uses", 0))
                m["fountain/heal_total"] = float(s.get("heal_total", 0))
                m["fountain/depletion_round"] = float(s.get("depleted_round", -1))
                m["fountain/use_gini"] = _gini(uses)
                for aid, e in self.agents.items():
                    m[f"fountain/uses/{aid}"] = float(s.get(f"use:{e}", 0))
            elif o.hook == "scn_shrine":
                don = [float(s.get(f"don:{e}", 0)) for e in self.agents.values()]
                m["shrine/pool"] = float(s.get("pool", 0))
                m["shrine/warded"] = 1.0 if s.get("warded") else 0.0
                m["shrine/ward_round"] = float(s.get("ward_round", -1))
                m["shrine/donors"] = float(sum(1 for d in don if d > 0))
                m["shrine/free_riders"] = float(sum(1 for d in don if d <= 0))
                m["shrine/free_rider_index"] = float(sum(1 for d in don if d <= 0)) / max(1, len(don))
                m["shrine/donation_gini"] = _gini(don)
                for aid, e in self.agents.items():
                    m[f"shrine/donated/{aid}"] = float(s.get(f"don:{e}", 0))
            elif o.hook == "scn_winch":
                m["winch/progress"] = float(s.get("progress", 0))
                m["winch/opened"] = 1.0 if s.get("opened") else 0.0
                m["winch/opened_round"] = float(s.get("opened_round", -1))
                for aid, e in self.agents.items():
                    m[f"winch/cranks/{aid}"] = float(s.get(f"prog:{e}", 0))
            elif o.hook == "scn_relic":
                m["relic/taken"] = 1.0 if s.get("taken") else 0.0
                m["relic/taken_round"] = float(s.get("taken_round", -1))
                tb = int(s.get("taken_by", -1))
                m["relic/taken_by_faction"] = float(world.entities[tb].faction) if tb >= 0 else -1.0
            elif o.hook == "scn_coffer":
                m["coffer/lots_left"] = float(o.charges)
                m["coffer/opened"] = 1.0 if s.get("opened") else 0.0
                takes = [float(s.get(f"take:{e}", 0)) for e in self.agents.values()]
                m["coffer/take_gini"] = _gini(takes)
                for aid, e in self.agents.items():
                    m[f"coffer/taken/{aid}"] = float(s.get(f"take:{e}", 0))
        return m
