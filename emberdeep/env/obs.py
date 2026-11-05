"""Observation construction (DESIGN §7.4).

:class:`ObsBuilder` turns the world (through a :class:`FrameTable`, an :class:`ActionContext` and the social
history) into the per-agent ``Dict`` observation.  The layout of every vector is declared in
:mod:`emberdeep.env.spaces`; nothing here hard-codes column numbers.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.entities import S
from emberdeep.core.types import (
    ClassId, Condition, DecisionKind, EntityKind, GameMode, ObjectiveKind, Relation, Terrain, TargetType,
)
from emberdeep.env import kernels as K
from emberdeep.env import spaces as SP

log = logging.getLogger(__name__)

_COND_SHIFTS = np.arange(SP.N_COND, dtype=np.int64)
_STATUS_SHIFTS = np.arange(SP.N_STATUS, dtype=np.int64)
_HIDDEN_BIT = 3                      # Status.HIDDEN == 1 << 3
_STATUS_BIT = {n: i for i, n in enumerate(SP.STATUS_NAMES)}
# Relation enum value -> relation one-hot column (self, ally, neutral, truce, hostile)
_REL_SLOT = np.array([4, 2, 3, 1], np.int8)
_LOG_GOLD = math.log1p(2000.0)
_CLASS_INDEX = {c: i for i, c in enumerate(SP.CLASS_NAMES)}
_AB_TAG_INDEX = {t: i for i, t in enumerate(SP.ABILITY_TAGS)}
_AB_KIND_INDEX = {k: i for i, k in enumerate(SP.ABILITY_KINDS)}
_EYE = np.eye(max(SP.MAX_FACTIONS + 1, 16), dtype=np.float32)


def _bits(values: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.float32)


def _gold_feat(g: float) -> float:
    return math.log1p(max(0.0, g)) / _LOG_GOLD


def _avg(expr: str) -> float:
    try:
        return float(D.average(expr))
    except Exception:  # noqa: BLE001
        return 0.0


class ObsBuilder:
    """Builds observations and the global state for one :class:`EmberdeepCore`."""

    def __init__(self, core):
        self.core = core
        cfg = core.cfg
        self.cfg = cfg
        self.V = core.layout.V
        self.fs = SP.self_features()
        self.fe = SP.entity_features(self.V)
        self.fa = SP.ability_features()
        self.R = cfg.view_radius
        self.W2 = 2 * self.R + 1
        self.vocab = SP.vocab()
        self._ab_cache: dict[int, tuple] = {}
        self._objective_cache: dict[int, np.ndarray] = {}
        self._zone_cache: dict[int, tuple] = {}
        self._static = None
        self._reach_cache: dict[int, tuple] = {}
        self._range_cache: dict[int, tuple] = {}
        self._vocab_arr = None
        self._row_cache: dict = {}
        self._stack_cache: dict = {}
        self._dyn = None
        self._dyn_key = None
        self._dyn_idx = [SP.MAP_INDEX[n] for n in ("opaque", "door_closed", "door_open")]
        self._sl_self = {k: v for k, v in self.fs.slices.items()}
        self._ix = self.fs.index
        self._ex = self.fe.index
        self._exs = self.fe.slices
        self._fdt = cfg.float_dtype
        self._nfe = len(self.fe)
        self._econsts = self._entity_consts()
        self._hp_mode = {"bucket": 0, "exact": 1, "hidden": 2}[cfg.enemy_hp]
        self._no_ack = np.zeros(64, bool)

    # ------------------------------------------------------------------------------------ per-episode
    def reset(self) -> None:
        w = self.core.world
        st = w.map.static
        H, W = st.height, st.width
        self.H, self.W = H, W
        self._ab_cache.clear()
        self._objective_cache.clear()
        self._zone_cache.clear()
        self._reach_cache.clear()
        self._range_cache.clear()
        self._vocab_arr = None
        self._row_cache = {}
        self._stack_cache = {}
        self._dyn_key = None
        door = np.zeros((H, W), bool)
        for d in w.objects:
            if d.kind == int(EntityKind.DOOR):
                door[d.y, d.x] = True
        self._static = {
            "difficult": st.difficult.astype(np.float32),
            "hazard": st.lava.astype(np.float32),
            "water": st.water.astype(np.float32),
            "pit": st.pit.astype(np.float32),
            "cover": (st.half_cover | st.tq_cover).astype(np.float32),
            "door_any": door,
            "exit": st.exit,
        }

    def objective_mask(self, faction: int) -> np.ndarray:
        m = self._objective_cache.get(faction)
        if m is not None:
            return m
        w = self.core.world
        m = np.zeros((self.H, self.W), np.float32)
        m[w.map.static.exit] = 1.0
        for st in w.objectives:
            sp = st.spec
            if st.faction_idx and faction not in st.faction_idx:
                continue
            if sp.kind in (ObjectiveKind.REACH_ZONE, ObjectiveKind.ESCAPE, ObjectiveKind.COLLECT):
                for name in sp.targets:
                    for (x, y) in w.regions.get(name, ()):
                        if 0 <= x < self.W and 0 <= y < self.H:
                            m[y, x] = 1.0
        self._objective_cache[faction] = m
        return m

    # ------------------------------------------------------------------------------------ public
    def build(self, agent: str, ctx, frame) -> dict:
        core = self.core
        cfg = self.cfg
        lay = core.layout
        obs: dict = {}
        obs["self"] = self._self_vec(agent, ctx, frame)
        ent, emask, eids = self._entities(agent, ctx, frame)
        obs["entities"] = ent
        obs["entity_mask"] = emask
        obs["entity_ids"] = eids
        ab, abid = self._abilities(agent, ctx)
        obs["abilities"] = ab
        obs["ability_ids"] = abid
        obs["map"] = self._map(agent, ctx, frame)
        if lay.V > 0:
            obs["messages"] = self._messages()
        obs["action_mask"] = ctx.action_mask
        obs["ability_mask"] = ctx.ability_mask
        obs["target_mask"] = ctx.target_mask
        if cfg.include_state_in_obs:
            obs["state"] = self.state(frame)
        return obs

    # ------------------------------------------------------------------------------------ self
    def _self_vec(self, agent: str, ctx, frame) -> np.ndarray:
        core = self.core
        w = core.world
        c = w.entities[core.agent_eid[agent]]
        ix = self._ix
        sl = self._sl_self
        out = np.zeros(len(self.fs), np.float32)
        maxhp = float(max(1, c.max_hp))
        out[ix["hp_frac"]] = c.hp / maxhp
        out[ix["temp_hp"]] = min(2.0, c.temp_hp / maxhp)
        out[ix["hp_abs"]] = c.hp / 100.0
        out[ix["max_hp"]] = c.max_hp / 100.0
        out[ix["ac"]] = R.compute_ac(w, c) / 30.0
        try:
            speed = R.base_speed(w, c)
        except Exception:  # noqa: BLE001
            speed = c.speeds.get("walk", 30)
        out[ix["speed"]] = speed / 60.0
        active = w.active == c.eid and not w.done
        ts = c.ts
        if active:
            out[ix["movement_left"]] = ts.movement_left / 60.0
            out[ix["action_avail"]] = 0.0 if ts.action_used else 1.0
            out[ix["bonus_avail"]] = 0.0 if ts.bonus_used else 1.0
            out[ix["free_interaction_avail"]] = 0.0 if ts.free_used else 1.0
            out[ix["attacks_left"]] = ts.attacks_left / 4.0
        out[ix["reaction_avail"]] = 1.0 if c.reaction_available else 0.0
        out[ix["pos_x"]] = c.x / max(1, self.W)
        out[ix["pos_y"]] = c.y / max(1, self.H)
        out[ix["altitude"]] = c.z / 12.0
        out[sl["cond"]] = _bits(np.array([c.cond_mask], np.int64), _COND_SHIFTS)[0]
        out[sl["status"]] = _bits(np.array([c.status], np.int64), _STATUS_SHIFTS)[0]
        out[ix["exhaustion"]] = c.exhaustion / 6.0
        out[ix["concentrating"]] = 1.0 if c.concentration is not None else 0.0
        out[ix["death_save_successes"]] = c.death_succ / 3.0
        out[ix["death_save_failures"]] = c.death_fail / 3.0
        s0 = sl["slots"].start
        for i in range(9):
            v = c.spell_slots[i]
            if v:
                out[s0 + i] = min(2.0, v / 4.0)
        out[ix["pact_slots"]] = c.pact_slots / 4.0
        out[ix["pact_level"]] = c.pact_level / 5.0
        r0 = sl["res"].start
        for i, rid in enumerate(SP.RESOURCE_IDS):
            mx = c.resource_max.get(rid) if rid != "other" else None
            if mx:
                out[r0 + i] = c.resources.get(rid, 0) / max(1, mx)
            else:
                out[r0 + i] = -1.0
        if "other" in SP.RESOURCE_IDS:
            others = [k for k in c.resource_max if k not in SP.RESOURCE_IDS]
            if others:
                tot = sum(c.resource_max[k] for k in others)
                out[r0 + SP.RESOURCE_IDS.index("other")] = sum(c.resources.get(k, 0) for k in others) / max(1, tot)
        out[ix["gold"]] = _gold_feat(c.gold)
        out[ix["potions"]] = sum(i.qty for i in c.inventory if getattr(i.def_, "heal", None)) / 3.0
        if c.is_pc and c.class_levels:
            primary = max(c.class_levels.items(), key=lambda kv: kv[1])[0]
            ci = _CLASS_INDEX.get(primary)
            if ci is not None:
                out[sl["class"].start + ci] = 1.0
        else:
            out[ix["is_monster"]] = 1.0
        out[ix["level"]] = (c.level if c.is_pc else c.cr) / 10.0
        for i in range(6):
            out[sl["mod"].start + i] = c.mods[i] / 5.0
        out[ix["mode_combat"]] = 1.0 if w.mode == int(GameMode.COMBAT) else 0.0
        out[ix["round"]] = w.round / max(1, w.rules.max_rounds)
        out[ix["is_decider"]] = 1.0 if ctx.is_decider else 0.0
        dk = 0
        if ctx.decision is not None:
            dk = int(ctx.decision.kind)
        out[sl["decision"].start + min(dk, len(SP.DECISION_NAMES) - 1)] = 1.0
        out[ix["rest_vote_self"]] = 1.0 if c.rest_vote else 0.0
        mates = [m for m in w.creatures if m.faction == c.faction and m.present and not m.status & S.DEAD]
        if mates:
            out[ix["rest_vote_frac"]] = sum(1 for m in mates if m.rest_vote) / len(mates)
        if c.faction < SP.MAX_FACTIONS:
            out[sl["faction"].start + c.faction] = 1.0
        nf = len(w.factions)
        if w.truce_offers:
            out[ix["truce_offer_incoming"]] = 1.0 if any(w.has_truce_offer(f, c.faction) for f in range(nf)
                                                          if f != c.faction) else 0.0
            out[ix["truce_offer_outgoing"]] = 1.0 if any(w.has_truce_offer(c.faction, f) for f in range(nf)
                                                          if f != c.faction) else 0.0
        out[ix["agent_index"]] = (core.agent_index[agent] + 1) / max(1, core.n_agents)
        np.clip(out, -SP.FEATURE_CLIP, SP.FEATURE_CLIP, out=out)
        return out.astype(self._fdt, copy=False)

    # ------------------------------------------------------------------------------------ entities
    def _best_range(self, c, abil) -> int:
        hit = self._range_cache.get(c.eid)
        if hit is not None and hit[0] is abil:
            return hit[1]
        best = 1
        for ab in abil:
            if ab.kind in ("weapon", "monster", "spell") and ("attack" in ab.tags or "damage" in ab.tags
                                                            or "melee" in ab.tags or "ranged" in ab.tags):
                best = max(best, ab.range_sq, ab.long_range_sq if "ranged" in ab.tags else 0, ab.reach_sq)
        self._range_cache[c.eid] = (abil, best)
        return best

    def _vocab_ids(self, eids: np.ndarray) -> np.ndarray:
        """Entity vocabulary ids for entity ids (cached per episode)."""
        w = self.core.world
        n = len(w.entities)
        arr = self._vocab_arr
        if arr is None or len(arr) < n:
            new = np.full(max(n, 64) * 2, -1, np.int32)
            if arr is not None:
                new[:len(arr)] = arr
            arr = self._vocab_arr = new
        v = arr[eids]
        miss = np.nonzero(v < 0)[0]
        if miss.size:
            for j in miss.tolist():
                e = w.entities[int(eids[j])]
                arr[eids[j]] = self.vocab.entity_id(getattr(e, "template_id", ""), int(e.kind))
            v = arr[eids]
        return v

    def _ent_static(self, frame) -> np.ndarray:
        """Observer-independent entity columns for *all* rows of the frame (built once per world version)."""
        st = getattr(frame, "_ent_static", None)
        if st is not None:
            return st
        core = self.core
        w = core.world
        fe, exs, ex = self.fe, self._exs, self._ex
        n = frame.n
        st = np.zeros((n, SP.N_ENT_STATIC), np.float32)
        is_cre = frame.kind == int(EntityKind.CREATURE)
        isc = is_cre[:, None]
        cb, sb = frame.bit_tables()
        st[:, 0] = 1.0
        st[:, exs["kind"]] = _EYE[frame.kind.astype(np.int64)][:, :len(SP.KIND_NAMES)]
        st[:, exs["cond"]] = cb
        st[:, exs["status"]] = sb
        st[:, ex["size"]] = np.where(is_cre, frame.size_enum * 0.2, 0.0)
        st[:, exs["ctype"]] = _EYE[frame.ctype.astype(np.int64)][:, :SP.N_CTYPE] * isc
        st[:, ex["level"]] = frame.level * 0.1
        st[:, ex["is_agent"]] = frame.is_agent
        ai = frame.agent_idx.astype(np.float32)
        st[:, ex["agent_index"]] = np.where(ai >= 0, (ai + 1) / max(1, core.n_agents), 0.0)
        st[:, ex["is_active"]] = (frame.eid == w.active) & is_cre
        st[:, ex["initiative"]] = np.where(is_cre, frame.initiative, -1.0)
        st[:, ex["concentrating"]] = frame.conc
        st[:, ex["downed"]] = frame.downed
        st[:, ex["surrendered"]] = sb[:, _STATUS_BIT["surrendered"]]
        st[:, ex["fleeing"]] = sb[:, _STATUS_BIT["fleeing"]]
        fac = frame.faction.astype(np.int64)
        okf = is_cre & (fac >= 0) & (fac < SP.MAX_FACTIONS)
        st[:, exs["faction"]] = _EYE[np.where(okf, fac, SP.MAX_FACTIONS)][:, :SP.MAX_FACTIONS]
        frame._ent_static = st
        return st

    def _entity_consts(self):
        """Column / constant vectors handed to :func:`kernels.fill_entities` (built once)."""
        ex, exs = self._ex, self._exs
        C = np.zeros(28, np.int64)
        C[K.C_KIND] = exs["kind"].start
        C[K.C_COND] = exs["cond"].start
        C[K.C_STATUS] = exs["status"].start
        C[K.C_SIZE] = ex["size"]
        C[K.C_CTYPE] = exs["ctype"].start
        C[K.C_LEVEL] = ex["level"]
        C[K.C_ISAGENT] = ex["is_agent"]
        C[K.C_AGENTIDX] = ex["agent_index"]
        C[K.C_ISACTIVE] = ex["is_active"]
        C[K.C_INIT] = ex["initiative"]
        C[K.C_CONC] = ex["concentrating"]
        C[K.C_DOWNED] = ex["downed"]
        C[K.C_SURR] = ex["surrendered"]
        C[K.C_FLEE] = ex["fleeing"]
        C[K.C_FACTION] = exs["faction"].start
        C[K.C_REL] = exs["rel"].start
        C[K.C_DX] = ex["dx"]
        C[K.C_REACH] = ex["in_reach"]
        C[K.C_HP] = ex["hp_frac"]
        C[K.C_SOC] = ex["dmg_to_me"]
        C[K.C_HIDDEN] = ex["status.hidden"]
        C[K.C_NKIND] = len(SP.KIND_NAMES)
        C[K.C_NCOND] = SP.N_COND
        C[K.C_NSTATUS] = SP.N_STATUS
        C[K.C_NCTYPE] = SP.N_CTYPE
        C[K.C_SURR_BIT] = _STATUS_BIT["surrendered"]
        C[K.C_FLEE_BIT] = _STATUS_BIT["fleeing"]
        C[K.C_NFACTIONS] = SP.MAX_FACTIONS
        return C

    def _entities(self, agent: str, ctx, frame):
        core = self.core
        w = core.world
        cfg = self.cfg
        E = cfg.max_entities
        me = core.agent_eid[agent]
        cme = w.entities[me]
        out = np.zeros((E, self._nfe), np.float32)
        mask = np.zeros(E, np.int8)
        ids = np.zeros(E, np.int32)
        k = ctx.n_slots
        if k == 0 or ctx.slot_rows[0] < 0:
            return out.astype(self._fdt, copy=False), mask, ids
        r = ctx.slot_rows[:k]
        eids_s = ctx.slot_eids[:k]
        ex, exs = self._ex, self._exs
        mask[:k] = 1
        myf = cme.faction
        history = core.history
        known_ac = self._no_ack[:k] if len(self._no_ack) >= k else np.zeros(k, bool)
        ack = history.ac_known.get(myf)
        if ack:
            known_ac = np.array([e in ack for e in eids_s.tolist()], bool)
        abil = w.abilities(me)
        mode = self._hp_mode
        P = np.array([myf, len(w.factions), mode, core.n_agents, w.active, w.reach_of(cme),
                      self._best_range(cme, abil), me, 1 if cfg.social_features else 0, frame.nc], np.int64)
        need = np.zeros(k, np.int8)
        K.fill_entities(out, need, r, ctx.visible_rows, frame.kind, frame.x, frame.y, frame.z, frame.size,
                        frame.faction, frame.present, frame.hp, frame.max_hp, frame.ac, frame.cond, frame.status,
                        frame.ctype, frame.size_enum, frame.level, frame.conc, frame.agent_idx, frame.initiative,
                        frame.eid, frame.downed, frame.is_agent, w.relations, history.mat, known_ac, P,
                        self._econsts, K.REL_SLOT)
        nz = np.flatnonzero(need)
        if nz.size:
            c_cov = ex["in_reach"] + 3
            for j in nz.tolist():
                if need[j] == 1:
                    out[j, c_cov] = int(w.cover(me, int(eids_s[j]))) * (1.0 / 3.0)
                else:
                    self._object_row(out[j], w.entities[int(eids_s[j])])
                    np.clip(out[j], -SP.FEATURE_CLIP, SP.FEATURE_CLIP, out=out[j])
        if w.truce_offers:
            kinds = frame.kind[r]
            fac = frame.faction[r].astype(np.int64)
            relc = out[:k, exs["rel"].start:exs["rel"].start + 2].sum(axis=1)
            foe = (kinds == int(EntityKind.CREATURE)) & (relc == 0)
            for j in np.nonzero(foe & (fac >= 0))[0].tolist():
                f = int(fac[j])
                out[j, ex["truce_offer_in"]] = 1.0 if w.has_truce_offer(f, myf) else 0.0
                out[j, ex["truce_offer_out"]] = 1.0 if w.has_truce_offer(myf, f) else 0.0
        say = history.say
        if self.V and say:
            s0 = exs["said"].start
            for spk, (tok, rd) in say.items():
                if not (0 <= tok < self.V):
                    continue
                js = np.flatnonzero(eids_s == spk)
                if js.size == 0:
                    continue
                j = int(js[0])
                if frame.kind[r[j]] != 0:
                    continue
                if w.hears(cme, w.entities[int(spk)]):
                    out[j, s0 + tok] = 0.5 ** max(0, w.round - rd)
        ids[:k] = self._vocab_ids(eids_s)
        # the kernel already clipped, but the SAY decay and cover columns are inside the range by construction
        return out.astype(self._fdt, copy=False), mask, ids

    def _entities_ref(self, agent: str, ctx, frame):
        core = self.core
        w = core.world
        cfg = self.cfg
        E = cfg.max_entities
        me = core.agent_eid[agent]
        cme = w.entities[me]
        fe = self.fe
        out = np.zeros((E, len(fe)), np.float32)
        mask = np.zeros(E, np.int8)
        ids = np.zeros(E, np.int32)
        k = ctx.n_slots
        if k == 0 or ctx.slot_rows[0] < 0:
            return out.astype(self._fdt, copy=False), mask, ids
        r = ctx.slot_rows[:k]
        eids_s = ctx.slot_eids[:k]
        ex, exs = self._ex, self._exs
        mask[:k] = 1
        out[:k, :SP.N_ENT_STATIC] = self._ent_static(frame)[r]
        is_cre = frame.kind[r] == int(EntityKind.CREATURE)
        fac = frame.faction[r].astype(np.int64)
        myf = cme.faction
        # --- relation to the observer
        nfac = len(w.factions)
        rv = w.relations[myf, np.clip(fac, 0, max(nfac - 1, 0))].astype(np.int64) if nfac else np.ones(k, np.int64)
        rv = np.where(fac == myf, int(Relation.ALLIED), rv)
        rel_slot = np.where(is_cre, _REL_SLOT[rv], 2)
        rel_slot[0] = 0
        out[:k, exs["rel"]] = _EYE[rel_slot][:, :SP.N_RELATIONS]
        ally = (rel_slot <= 1) & is_cre
        foe = is_cre & ~ally
        out[np.nonzero(~ally)[0], ex["status.hidden"]] = 0.0                # hidden bit visible for allies only
        # --- geometry (dx, dy, dz, dist are adjacent columns)
        r0 = int(r[0])
        dist = frame.distance_from(r0)[r]
        c4 = ex["dx"]
        geo = np.empty((k, 4), np.float32)
        geo[:, 0] = (frame.x[r] - frame.x[r0]) * 0.05
        geo[:, 1] = (frame.y[r] - frame.y[r0]) * 0.05
        geo[:, 2] = (frame.z[r] - frame.z[r0]) * (1.0 / 6.0)
        np.clip(geo[:, :3], -2.0, 2.0, out=geo[:, :3])
        geo[:, 3] = np.clip(dist * 0.05, 0.0, 2.0)
        out[:k, c4:c4 + 4] = geo
        abil = w.abilities(me)
        vis = ctx.visible_rows[r].copy()
        vis[0] = True
        flags = np.zeros((k, 4), np.float32)                                # in_reach, in_range, has_los, cover
        flags[:, 0] = dist <= w.reach_of(cme)
        flags[:, 1] = dist <= self._best_range(cme, abil)
        flags[:, 2] = vis
        for j in np.nonzero(vis & foe & (dist > 1))[0].tolist():
            flags[j, 3] = int(w.cover(me, int(eids_s[j]))) * (1.0 / 3.0)
        c_f = ex["in_reach"]
        out[:k, c_f:c_f + 4] = flags
        # --- vitals (hp_frac, hp_known, hp_abs, ac, ac_known are adjacent)
        hp = frame.hp[r]
        frac = hp / frame.max_hp[r]
        mode = cfg.enemy_hp
        vit = np.zeros((k, 5), np.float32)
        if mode == "hidden":
            known_hp = ally
            hpf = frac
        else:
            known_hp = is_cre
            hpf = np.where(ally, frac, np.where(hp <= 0, 0.0, np.ceil(np.clip(frac, 0, 1) * 4.0) * 0.25)) \
                if mode == "bucket" else frac
        vit[:, 0] = np.where(known_hp, hpf, 0.0)
        vit[:, 1] = known_hp
        vit[:, 2] = np.where((ally | (mode == "exact")) & is_cre, hp * 0.01, 0.0)
        known_ac = ally
        ack = core.history.ac_known.get(myf)
        if ack:
            known_ac = ally | (np.fromiter((int(e) in ack for e in eids_s), bool, k) & is_cre)
        vit[:, 3] = np.where(known_ac, frame.ac[r] * (1.0 / 30.0), 0.0)
        vit[:, 4] = known_ac
        c_v = ex["hp_frac"]
        out[:k, c_v:c_v + 5] = vit
        if w.truce_offers:
            for j in np.nonzero(foe & (fac >= 0))[0].tolist():
                f = int(fac[j])
                out[j, ex["truce_offer_in"]] = 1.0 if w.has_truce_offer(f, myf) else 0.0
                out[j, ex["truce_offer_out"]] = 1.0 if w.has_truce_offer(myf, f) else 0.0
        # --- last SAY tokens
        say = core.history.say
        if self.V and say:
            s0 = exs["said"].start
            for j in np.nonzero(is_cre)[0].tolist():
                sv = say.get(int(eids_s[j]))
                if sv is None:
                    continue
                tok, rd = sv
                if 0 <= tok < self.V and w.hears(cme, w.entities[int(eids_s[j])]):
                    out[j, s0 + tok] = 0.5 ** max(0, w.round - rd)
        # --- social history (dmg_to_me .. betrayals are adjacent)
        if cfg.social_features:
            blk = core.history.social_block(eids_s, me, core.faction_eids(myf))
            m = foe | (rel_slot == 1)                                       # other creatures (not self, not objects)
            m[0] = False
            lim = SP.FEATURE_CLIP
            soc = np.empty((k, 6), np.float32)
            soc[:, 0] = blk[0, :, 0] * 0.02
            soc[:, 1] = blk[0, :, 1] * 0.02
            soc[:, 2] = blk[1, :, 0] * 0.02
            soc[:, 3] = blk[1, :, 1] * 0.02
            soc[:, 4] = blk[2, :, 0] * 0.01
            soc[:, 5] = (blk[3, :, 0] + blk[3, :, 1]) * (1.0 / 3.0)
            soc *= m[:, None]
            np.minimum(soc, lim, out=soc)
            c_s = ex["dmg_to_me"]
            out[:k, c_s:c_s + 6] = soc
        # --- objects
        if not is_cre.all():
            for j in np.nonzero(~is_cre)[0].tolist():
                self._object_row(out[j], w.entities[int(eids_s[j])])
        ids[:k] = self._vocab_ids(eids_s)
        np.clip(out, -SP.FEATURE_CLIP, SP.FEATURE_CLIP, out=out)
        return out.astype(self._fdt, copy=False), mask, ids

    def _object_row(self, row: np.ndarray, o) -> None:
        ex = self._ex
        k = int(o.kind)
        if k == int(EntityKind.DOOR):
            st = int(o.state)
            if st == 0:
                row[ex["obj_open"]] = 1
            elif st == 1:
                row[ex["obj_closed"]] = 1
            elif st == 2:
                row[ex["obj_closed"]] = 1
                row[ex["obj_locked"]] = 1
            else:
                row[ex["obj_broken"]] = 1
        elif k == int(EntityKind.CHEST):
            if o.open:
                row[ex["obj_open"]] = 1
            else:
                row[ex["obj_closed"]] = 1
                if o.locked:
                    row[ex["obj_locked"]] = 1
            row[ex["obj_value"]] = math.log1p(o.gold + sum(i.value_gp for i in o.items)) / math.log1p(1000)
        elif k == int(EntityKind.LOOT):
            row[ex["obj_value"]] = math.log1p(o.value_gp) / math.log1p(1000)
        elif k == int(EntityKind.LEVER):
            row[ex["obj_open"] if o.on else ex["obj_closed"]] = 1
        elif k == int(EntityKind.TRAP):
            row[ex["obj_broken"]] = 1.0 if (o.disarmed or o.triggered) else 0.0
            row[ex["obj_closed"]] = 0.0 if (o.disarmed or o.triggered) else 1.0
        elif k == int(EntityKind.ALTAR):
            row[ex["obj_charges"]] = o.charges / max(1, o.max_charges) if o.max_charges else 0.0

    # ------------------------------------------------------------------------------------ abilities
    def _ability_static(self, c, ab) -> np.ndarray:
        fa = self.fa
        ix = fa.index
        sl = fa.slices
        row = np.zeros(len(fa), np.float32)
        row[ix["exists"]] = 1.0
        ki = _AB_KIND_INDEX.get(ab.kind)
        if ki is not None:
            row[sl["kind"].start + ki] = 1.0
        row[sl["cost"].start + min(int(ab.cost), len(SP.COST_NAMES) - 1)] = 1.0
        row[sl["target"].start + min(int(ab.target), len(SP.TARGET_NAMES) - 1)] = 1.0
        row[ix["range"]] = min(ab.range_sq, 60) / 30.0
        row[ix["long_range"]] = min(ab.long_range_sq, 60) / 30.0
        row[ix["reach"]] = min(ab.reach_sq, 12) / 6.0
        if ab.area is not None:
            row[sl["area"].start + int(ab.area.shape)] = 1.0
            row[ix["area_size"]] = ab.area.size_ft / 5.0 / 12.0
        else:
            row[sl["area"].start] = 1.0
        row[ix["spell_level"]] = ab.spell_level / 9.0 if ab.spell_level >= 0 else -1.0
        dmg, dtypes, heal, save, dc, atk, conc = 0.0, (), 0.0, None, 0, 0, False
        src = ab.source
        name = type(src).__name__
        try:
            if name == "SpellDef":
                dmg = sum(_avg(d.dice) for d in src.damage)
                dtypes = tuple(int(d.type) for d in src.damage)
                heal = _avg(src.healing) if src.healing else 0.0
                if src.healing and src.heal_add_mod and c.spell_ability is not None:
                    heal += c.mods[int(c.spell_ability)]
                conc = bool(src.concentration)
                sc_dc, sc_atk = self._spell_numbers(c)
                if src.save is not None:
                    save = int(src.save.ability)
                    dc = src.save.dc if src.save.dc else sc_dc
                if src.attack is not None:
                    atk = sc_atk
            elif name == "AttackDef":
                dmg = sum(_avg(d.dice) for d in src.damage)
                dtypes = tuple(int(d.type) for d in src.damage)
                atk = src.to_hit if src.to_hit is not None else c.prof
            elif name == "SaveActionDef":
                dmg = sum(_avg(d.dice) for d in src.damage)
                dtypes = tuple(int(d.type) for d in src.damage)
                save = int(src.save.ability)
                dc = src.save.dc or c.monster_spell_dc or 8 + c.prof
            elif name == "WeaponDef":
                from emberdeep.core.abilities import weapon_stats
                _r, bonus, dice, _m, _rc, _rng, _mg = weapon_stats(
                    self.core.world, c, ab.data["item"], src, bool(ab.data.get("thrown")), bool(ab.data.get("offhand")))
                dmg = _avg(dice)
                dtypes = (int(src.damage_type),)
                atk = bonus
            elif name == "ItemDef":
                if src.heal:
                    heal = _avg(src.heal)
                if src.damage is not None:
                    dmg = _avg(src.damage.dice)
                    dtypes = (int(src.damage.type),)
        except Exception:  # noqa: BLE001
            log.debug("ability feature extraction failed for %s", ab.id, exc_info=True)
        row[ix["avg_damage"]] = min(dmg / 30.0, SP.FEATURE_CLIP)
        for t in dtypes:
            row[sl["dtype"].start + t] = 1.0
        row[ix["avg_healing"]] = min(heal / 20.0, SP.FEATURE_CLIP)
        row[ix["concentration"]] = 1.0 if conc else 0.0
        if save is not None:
            row[sl["save"].start + save] = 1.0
        row[ix["attack_bonus"]] = atk / 10.0
        row[ix["save_dc"]] = dc / 20.0
        for t in ab.tags:
            j = _AB_TAG_INDEX.get(t)
            if j is not None:
                row[sl["tag"].start + j] = 1.0
        return row

    @staticmethod
    def _spell_numbers(c) -> tuple[int, int]:
        if c.monster_spell_dc or c.monster_spell_atk:
            return c.monster_spell_dc, c.monster_spell_atk
        if c.spell_ability is not None:
            m = c.mods[int(c.spell_ability)]
            return 8 + c.prof + m, c.prof + m
        return 8 + c.prof, c.prof

    def _uses_left(self, c, ab) -> float:
        kind = ab.kind
        if kind == "spell":
            lvl = ab.spell_level
            if lvl <= 0:
                return -1.0
            n = sum(c.spell_slots[lvl - 1:]) + (c.pact_slots if c.pact_level >= lvl else 0)
            return n / 5.0
        if kind == "item":
            it = ab.data.get("item")
            if it is not None:
                return (it.charges if it.charges else it.qty) / 5.0
            return -1.0
        v = c.uses_left.get(ab.id)
        if v is None:
            v = c.resources.get(ab.id)
        if v is None:
            return -1.0
        return min(float(v), 25.0) / 5.0

    def _abilities(self, agent: str, ctx):
        core = self.core
        w = core.world
        lay = core.layout
        eid = core.agent_eid[agent]
        c = w.entities[eid]
        abil = w.abilities(eid)
        hit = self._ab_cache.get(eid)
        if hit is None or hit[0] is not abil or hit[3] != c.version:
            n = min(len(abil), lay.K)
            static = np.zeros((lay.K, len(self.fa)), np.float32)
            ids = np.zeros(lay.K, np.int32)
            cnum = (c.prof, tuple(c.mods), c.spell_ability, c.monster_spell_dc, c.monster_spell_atk)
            rc = self._row_cache
            for k in range(n):
                ab = abil[k]
                if ab.kind == "weapon":                      # depends on the carried item / fighting styles
                    static[k] = self._ability_static(c, ab)
                else:
                    key = (ab.id, ab.kind, int(ab.cost), int(ab.target), ab.range_sq, ab.long_range_sq, ab.reach_sq,
                           ab.spell_level, id(ab.source), ab.area, ab.tags, cnum)
                    row = rc.get(key)
                    if row is None:
                        row = rc[key] = self._ability_static(c, ab)
                    static[k] = row
                ids[k] = self.vocab.ability_id(ab.id)
            hit = (abil, static, ids, c.version)
            self._ab_cache[eid] = hit
        static = hit[1].copy()
        ix = self.fa.index
        n = min(len(abil), lay.K)
        static[:n, ix["available"]] = ctx.ability_mask[:n]
        ul = ix["uses_left"]
        if n:
            static[:n, ul] = [self._uses_left(c, ab) for ab in abil[:n]]
        static[n:, ul] = 0.0
        return static.astype(self._fdt, copy=False), hit[2].copy()

    # ------------------------------------------------------------------------------------ messages
    def _messages(self) -> np.ndarray:
        core = self.core
        w = core.world
        V = core.layout.V
        out = np.zeros((core.n_agents, V), np.float32)
        say = core.history.say
        for a, name in enumerate(core.agents):
            sv = say.get(core.agent_eid[name])
            if sv is None:
                continue
            tok, rd = sv
            if 0 <= tok < V:
                out[a, tok] = 0.5 ** max(0, w.round - rd)
        return out.astype(self._fdt, copy=False)

    # ------------------------------------------------------------------------------------ map
    def _map(self, agent: str, ctx, frame) -> np.ndarray:
        core = self.core
        w = core.world
        cfg = self.cfg
        R, W2 = self.R, self.W2
        eid = core.agent_eid[agent]
        c = w.entities[eid]
        H, Wd = self.H, self.W
        out = np.zeros((SP.N_MAP_CHANNELS, W2, W2), np.float32)
        x0, y0 = c.x - R, c.y - R
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(Wd, c.x + R + 1), min(H, c.y + R + 1)
        if sx1 <= sx0 or sy1 <= sy0:
            return self._finish_map(out)
        ox0, oy0 = sx0 - x0, sy0 - y0
        ox1, oy1 = ox0 + (sx1 - sx0), oy0 + (sy1 - sy0)
        M = SP.MAP_INDEX
        st = self._static
        m = w.map
        win = (slice(sy0, sy1), slice(sx0, sx1))

        def put(name: str, arr) -> None:
            out[M[name], oy0:oy1, ox0:ox1] = arr[win]

        explored = core.update_explored(agent)
        vis = core.visible_for(agent)
        if cfg.fog_of_war_map:
            known = (explored | vis)[win]
            out[M["in_bounds"], oy0:oy1, ox0:ox1] = 1.0
            opq = m.opaque
            dc = m.door_closed
            for name, arr in (("opaque", opq), ("door_closed", dc), ("door_open", st["door_any"] & ~dc),
                              ("difficult", st["difficult"]), ("hazard", st["hazard"]), ("water", st["water"]),
                              ("pit", st["pit"]), ("cover", st["cover"]), ("objective", self.objective_mask(c.faction))):
                out[M[name], oy0:oy1, ox0:ox1] = arr[win] * known
        else:
            # static channels (one block copy) + door/opacity channels cached per opacity version
            stack, idx = self._static_stack(c.faction)
            out[idx, oy0:oy1, ox0:ox1] = stack[:, sy0:sy1, sx0:sx1]
            dyn = self._dyn_stack(m)
            out[self._dyn_idx, oy0:oy1, ox0:ox1] = dyn[:, sy0:sy1, sx0:sx1]
        out[M["visible"], oy0:oy1, ox0:ox1] = vis[win]
        out[M["explored"], oy0:oy1, ox0:ox1] = explored[win]
        lg = w.light_grid()
        out[M["light"], oy0:oy1, ox0:ox1] = lg[win] * 0.5
        # --- creatures / objects from the entity slots
        myf = c.faction
        rows = ctx.slot_rows
        ch_ally, ch_enemy, ch_neu, ch_obj = M["allies"], M["enemies"], M["neutrals"], M["objects"]
        ch_thr = M["threat"]
        relations = w.relations
        nf = len(w.factions)
        for s in range(1, len(rows)):
            r = int(rows[s])
            if r < 0:
                continue
            e = frame.ents[r]
            if frame.kind[r] == int(EntityKind.CREATURE):
                if not frame.present[r]:
                    continue
                f = e.faction
                if f == myf:
                    ch, hostile = ch_ally, False
                else:
                    rv = int(relations[myf, f]) if f < nf else 1
                    if rv == int(Relation.ALLIED):
                        ch, hostile = ch_ally, False
                    elif rv == int(Relation.HOSTILE):
                        ch, hostile = ch_enemy, True
                    else:
                        ch, hostile = ch_neu, False
                if not ctx.visible_rows[r] and f != myf:
                    continue
                sz = e.size_sq
                self._mark(out, ch, e.x - x0, e.y - y0, sz)
                if hostile and ctx.visible_rows[r]:
                    rch = w.reach_of(e)
                    self._mark(out, ch_thr, e.x - x0 - rch, e.y - y0 - rch, sz + 2 * rch)
            else:
                if e.kind == int(EntityKind.DOOR):
                    continue
                self._mark(out, ch_obj, e.x - x0, e.y - y0, 1)
        # self marker goes on the allies channel? no: self is the window centre by construction
        # --- zones
        zh, zb = M["zone_harmful"], M["zone_helpful"]
        for z in w.zones:
            arr = self._zone_cells(z)
            if arr.size == 0:
                continue
            rx = arr[:, 0] - x0
            ry = arr[:, 1] - y0
            ok = (rx >= 0) & (rx < W2) & (ry >= 0) & (ry < W2)
            if ok.any():
                out[zh if z.harmful else zb, ry[ok], rx[ok]] = 1.0
        # --- reachable cells (decider only)
        if ctx.is_decider and ctx.decision is None:
            self._reachable(out, c, frame, x0, y0, M["reachable"])
        return self._finish_map(out)

    def _static_stack(self, faction: int):
        hit = self._stack_cache.get(faction)
        if hit is None:
            M, st = SP.MAP_INDEX, self._static
            names = ["in_bounds", "difficult", "hazard", "water", "pit", "cover", "objective"]
            arrs = [np.ones((self.H, self.W), np.float32), st["difficult"], st["hazard"], st["water"], st["pit"],
                    st["cover"], self.objective_mask(faction)]
            hit = (np.ascontiguousarray(np.stack(arrs)), [M[n] for n in names])
            self._stack_cache[faction] = hit
        return hit

    def _dyn_stack(self, m) -> np.ndarray:
        key = (m.opaque_version, m.move_version)
        if self._dyn_key != key:
            dc = m.door_closed
            self._dyn = np.stack([m.opaque, dc, self._static["door_any"] & ~dc]).astype(np.float32)
            self._dyn_key = key
        return self._dyn

    def _finish_map(self, out: np.ndarray) -> np.ndarray:
        if self.cfg.map_dtype == "uint8":
            return (out * 255.0 + 0.5).astype(np.uint8)
        if self.cfg.map_dtype == "auto":
            return out.astype(self._fdt, copy=False)
        return out.astype(np.float16 if self.cfg.map_dtype == "float16" else np.float32, copy=False)

    def _mark(self, out, ch: int, rx: int, ry: int, size: int) -> None:
        W2 = self.W2
        x0, y0 = max(0, rx), max(0, ry)
        x1, y1 = min(W2, rx + size), min(W2, ry + size)
        if x1 > x0 and y1 > y0:
            out[ch, y0:y1, x0:x1] = 1.0

    def _zone_cells(self, z) -> np.ndarray:
        hit = self._zone_cache.get(z.zid)
        if hit is not None and hit[0] is z.cells:
            return hit[1]
        arr = np.asarray(list(z.cells), np.int32).reshape(-1, 2)
        self._zone_cache[z.zid] = (z.cells, arr)
        return arr

    def _reachable(self, out, c, frame, x0, y0, ch) -> None:
        w = self.core.world
        ts = c.ts
        ml = ts.movement_left
        if ml < 5:
            return
        m = w.map
        mode = "fly" if c.z > 0 else ("swim" if c.speeds.get("swim") else "walk")
        cost = m.cost_grid(mode, c.size_sq) * 5.0
        hostile_rows = np.nonzero((frame.kind == int(EntityKind.CREATURE)) & frame.present
                                  & (frame.faction >= 0) & (frame.faction != c.faction))[0]
        if hostile_rows.size and c.z == 0:
            cost = cost.copy()
            relations = w.relations
            for r in hostile_rows.tolist():
                e = frame.ents[r]
                if e is c or e.z > 0 or int(relations[c.faction, e.faction]) != int(Relation.HOSTILE):
                    continue
                s = e.size_sq
                cost[max(0, e.y - (c.size_sq - 1)):e.y + s, max(0, e.x - (c.size_sq - 1)):e.x + s] = np.inf
        rule = 1 if w.rules.diagonal_rule == "5-10-5" else 0
        cost32 = cost.astype(np.float32)
        fld = getattr(w, "_ap_field", None)
        if fld is not None and fld[0] == c.eid and fld[1] == c.x and fld[2] == c.y and fld[3] == rule \
                and np.array_equal(fld[4], cost32):
            dist = fld[5]                              # same costs: the unbounded field agrees with the bounded one
        else:
            try:
                from emberdeep.core import _geom_fallback as GF
                dist = GF.geo.dijkstra(cost32, c.x, c.y, float(ml), rule)
            except Exception:  # noqa: BLE001
                log.debug("dijkstra failed", exc_info=True)
                return
        reach = np.isfinite(dist) & (dist <= ml)
        R, W2 = self.R, self.W2
        H, Wd = self.H, self.W
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(Wd, c.x + R + 1), min(H, c.y + R + 1)
        out[ch, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = reach[sy0:sy1, sx0:sx1]

    # ------------------------------------------------------------------------------------ global state
    def state(self, frame) -> np.ndarray:
        """Global CTDE state vector (fixed creature order + downsampled full map + scenario progress)."""
        core = self.core
        w = core.world
        N = core.state_entities
        nf = len(SP.STATE_CREATURE_FIELDS)
        cre = np.zeros((N, nf), np.float32)
        order = core.state_order()
        cix = {n: i for i, (n, _d) in enumerate(SP.STATE_CREATURE_FIELDS)}
        for slot, eid in enumerate(order[:N]):
            c = w.entities[eid]
            row = cre[slot]
            row[cix["present"]] = 1.0 if c.present else 0.0
            row[cix["hp_frac"]] = c.hp / max(1, c.max_hp)
            row[cix["hp_abs"]] = c.hp / 100.0
            row[cix["ac"]] = R.compute_ac(w, c) / 30.0
            row[cix["pos_x"]] = c.x / max(1, self.W)
            row[cix["pos_y"]] = c.y / max(1, self.H)
            row[cix["z"]] = c.z / 12.0
            row[cix["size"]] = c.size / 5.0
            ai = core.agent_index_of_eid.get(eid)
            row[cix["is_agent"]] = 1.0 if ai is not None else 0.0
            row[cix["agent_index"]] = (ai + 1) / max(1, core.n_agents) if ai is not None else 0.0
            row[cix["is_active"]] = 1.0 if w.active == eid and not w.done else 0.0
            row[cix["downed"]] = 1.0 if c.downed else 0.0
            row[cix["dead"]] = 1.0 if c.status & S.DEAD else 0.0
            row[cix["escaped"]] = 1.0 if c.status & S.ESCAPED else 0.0
            row[cix["concentrating"]] = 1.0 if c.concentration is not None else 0.0
            row[cix["gold"]] = _gold_feat(c.gold)
            row[cix["wealth"]] = _gold_feat(c.wealth)
            row[cix["level"]] = (c.level if c.is_pc else c.cr) / 10.0
            row[cix["slots"]] = min(2.0, (sum(c.spell_slots) + c.pact_slots) / 12.0)
            row[cix["movement_left"]] = c.ts.movement_left / 60.0 if w.active == eid else 0.0
        grid = self._state_grid(frame)
        extra = np.zeros(SP.state_extra_dim(), np.float32)
        extra[0] = w.round / max(1, w.rules.max_rounds)
        extra[1] = 1.0 if w.mode == int(GameMode.COMBAT) else 0.0
        extra[2] = core.env_steps / max(1, core.max_env_steps)
        alive = sum(1 for a in core.agents if core.is_live(a))
        extra[3] = alive / max(1, core.n_agents)
        o = 4
        for i, st in enumerate(w.objectives[:SP.MAX_STATE_OBJECTIVES]):
            extra[o + 2 * i] = 1.0 if st.done else 0.0
            extra[o + 2 * i + 1] = 1.0 if st.failed else 0.0
        o += 2 * SP.MAX_STATE_OBJECTIVES
        nf_ = min(len(w.factions), SP.MAX_FACTIONS)
        if nf_:
            rel = np.zeros((SP.MAX_FACTIONS, SP.MAX_FACTIONS), np.float32)
            rel[:nf_, :nf_] = (w.relations[:nf_, :nf_] + 1) / 4.0
            extra[o:o + SP.MAX_FACTIONS ** 2] = rel.ravel()
        o += SP.MAX_FACTIONS ** 2
        for i, a in enumerate(core.agents[:SP.MAX_FACTIONS]):
            extra[o + i] = _gold_feat(w.entities[core.agent_eid[a]].gold)
        vec = np.concatenate([cre.ravel(), grid.ravel(), extra])
        np.clip(vec, -SP.FEATURE_CLIP, SP.FEATURE_CLIP, out=vec)
        return vec.astype(self._fdt, copy=False)

    def _state_grid(self, frame) -> np.ndarray:
        core = self.core
        w = core.world
        st = w.map.static
        H, Wd = self.H, self.W
        G = SP.STATE_GRID
        chans = np.zeros((len(SP.STATE_MAP_CHANNELS), H, Wd), np.float32)
        chans[0] = w.map.opaque | st.blocks_move
        chans[1] = st.hazard | st.water
        agent_f = {w.entities[e].faction for e in core.agent_eid.values()}
        for r in range(frame.nc):
            e = frame.ents[r]
            if not e.present or e.faction < 0:
                continue
            ch = 2 if e.eid in core.agent_index_of_eid else 3
            if ch == 3 and e.faction in agent_f and all(
                    w.relation_f(e.faction, f) != Relation.HOSTILE for f in agent_f if f != e.faction):
                ch = 2
            s = e.size_sq
            chans[ch, e.y:e.y + s, e.x:e.x + s] = 1.0
        for r in range(frame.nc, frame.n):
            e = frame.ents[r]
            if e.kind != int(EntityKind.DOOR):
                chans[4, e.y, e.x] = 1.0
        chans[5] = st.exit
        for f in agent_f:
            chans[5] = np.maximum(chans[5], self.objective_mask(f))
        ex = np.zeros((H, Wd), bool)
        for a in core.agents:
            ex |= core.explored.get(a, ex)
        chans[6] = ex
        # downsample to GxG by block averaging
        ys = np.linspace(0, H, G + 1).astype(int)
        xs = np.linspace(0, Wd, G + 1).astype(int)
        cs = np.zeros((chans.shape[0], H + 1, Wd + 1), np.float32)
        cs[:, 1:, 1:] = chans.cumsum(1).cumsum(2)
        y0, y1 = ys[:-1], np.maximum(ys[1:], ys[:-1] + 1)
        x0, x1 = xs[:-1], np.maximum(xs[1:], xs[:-1] + 1)
        y1 = np.minimum(y1, H)
        x1 = np.minimum(x1, Wd)
        tot = (cs[:, y1][:, :, x1] - cs[:, y0][:, :, x1] - cs[:, y1][:, :, x0] + cs[:, y0][:, :, x0])
        area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
        return tot / np.maximum(area, 1.0)
