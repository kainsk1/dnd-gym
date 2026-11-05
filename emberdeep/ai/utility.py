"""Utility scoring of (ability, target / point) candidates (DESIGN §8).

``survey(sc, prof)`` walks the creature's *currently usable* abilities (``world.ability_mask``), scores every legal
target with ``world.estimate`` and class-aware heuristics, and returns

* a list of :class:`Cand` (immediately executable acts with a score in expected hit points), and
* a ``potential`` table telling the movement planner what each foe would be worth if it were in melee reach / range.

Scoring terms
-------------
damage      ``min(expected damage, target hp)``
kill        ``P(kill) * (1.6 * threat + 2)``; threat = expected damage per round of the target against us
focus       ``focus_w * (1 - hp fraction) * threat`` (wounded targets first -> emergent focus fire)
priority    casters / healers / leaders (``caster_w``), foes next to squishy allies (``guard_w``), isolated foes
control     ``P(fail save) * condition value * threat * duration``
healing     ``min(heal, missing hp) * urgency``; reviving a dying ally counts triple
buff        table value per recipient (``abinfo.BUFF_VALUE``) scaled by the number of recipients
risk        friendly fire in an area (allies, truce / neutral parties), concentration replacement, slot cost
"""
from __future__ import annotations

from emberdeep.ai.abinfo import BUFF_VALUE, COND_VALUE, ctype_ok, heal_amount, info
from emberdeep.ai.base import mem, turn_mem
from emberdeep.core import dice as D
from emberdeep.core import resolution as R
from emberdeep.core.abilities import N_UNIVERSAL, best_potion
from emberdeep.core.actions import Act, ActKind
from emberdeep.core.entities import S
from emberdeep.core.types import AreaShape, TargetType

_SQUISHY_AC = 13
_CASTER_TAGS = ("caster", "healer", "leader", "priest")


class Cand:
    """A scored candidate act."""
    __slots__ = ("score", "act", "tag")

    def __init__(self, score: float, act: Act, tag: str = ""):
        self.score = score
        self.act = act
        self.tag = tag

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Cand {self.score:.2f} {self.tag}>"


class Pot:
    """What a foe is worth if reachable in melee (``mval`` within ``reach``) or at range (``rval`` within ``rng``)."""
    __slots__ = ("f", "mval", "reach", "rval", "rng", "rngl")

    def __init__(self, f):
        self.f = f
        self.mval = 0.0
        self.reach = 1
        self.rval = 0.0
        self.rng = 0
        self.rngl = 0


class Survey:
    __slots__ = ("cands", "pot", "heal_pot", "best_attack", "has_attack_now", "use_stale", "best_ed")

    def __init__(self):
        self.cands: list[Cand] = []
        self.pot: dict[int, Pot] = {}
        self.heal_pot: list[tuple] = []        # (ally, value, range in squares)
        self.best_attack = 0.0
        self.has_attack_now = False
        self.use_stale = True
        self.best_ed: dict = {}


def ab_act(k: int, target: int = -1, point=None) -> Act:
    return Act(ActKind.ABILITY, k, target, point)


# ---------------------------------------------------------------------------------------------------- helpers
def is_squishy(w, a) -> bool:
    return a.max_hp <= 26 or R.compute_ac(w, a) <= _SQUISHY_AC


def has_feature(c, fid: str) -> bool:
    for h in c.hooks:
        if h.ID == fid:
            return True
    return False


def slot_cost(prof, inf) -> float:
    """Opportunity cost (hp) of burning a spell slot / limited resource."""
    if inf.slot and inf.level > 0:
        return prof.slot_w * (0.9 + 0.8 * inf.level)
    if inf.limited and inf.recharge == 0:
        return 0.8 * prof.slot_w
    return 0.0


def conc_penalty(sc, prof, inf, v: float) -> float:
    """Value lost by replacing the concentration spell we are already holding."""
    if inf.conc and sc.c.status & S.CONCENTRATING:
        cur = mem(sc.c).get("conc_val", 5.0)
        return prof.conc_hold * cur
    return 0.0


def sneak_bonus(sc, inf, ab, f) -> float:
    """Expected extra sneak-attack damage if a hit on ``f`` would qualify (ally adjacent to ``f``, not yet used)."""
    c = sc.c
    if c.ts.sneak_used or not has_feature(c, "sneak_attack"):
        return 0.0
    if inf.role != "attack" or ab.kind != "weapon":
        return 0.0
    wd = ab.data.get("wd")
    from emberdeep.core.types import WeaponProperty as WP
    if wd is None or not (inf.ranged or wd.properties & WP.FINESSE):
        return 0.0
    w = sc.w
    ok = False
    for a in sc.allies:
        if not a.cond_mask & 0x2A0 and w.distance(a.eid, f.eid) <= 1:
            ok = True
            break
    if not ok and not (f.cond_mask & 0x2A0):
        return 0.0
    lv = c.class_levels.get("rogue", c.level)
    return 3.5 * ((lv + 1) // 2) * 0.75


def attack_value(sc, prof, inf, f, ph: float, ed: float, ab=None) -> float:
    """Value of an attack-like effect with hit chance ``ph`` and expected damage ``ed`` on foe ``f``."""
    hp = f.hp + f.temp_hp
    if ab is not None:
        ed += ph * sneak_bonus(sc, inf, ab, f)
    dmg = min(ed, float(hp))
    hit_dmg = ed / ph if ph > 0.02 else 0.0
    if hit_dmg >= hp:
        kp = ph
    elif hit_dmg >= 0.75 * hp:
        kp = ph * 0.35
    else:
        kp = 0.0
    if f.status & (S.DYING | S.STABLE):
        return (dmg * 0.5 + 1.0) * prof.finish_downed
    thr = sc.threat(f)
    v = dmg + prof.kill_w * kp * (thr * 1.6 + 2.0)
    hpf = f.hp / max(1, f.max_hp)
    v += prof.focus_w * (1.0 - hpf) * thr * min(1.0, ph + 0.2)
    tags = f.tags
    if prof.caster_w and tags and any(t in tags for t in _CASTER_TAGS):
        v += prof.caster_w * thr
    if f.spells or f.cantrips:
        v += 0.5 * prof.caster_w * thr
    w = sc.w
    if prof.guard_w:
        for a in sc.allies:
            if sc.dist2(a, f) <= 2 and is_squishy(w, a):
                v += prof.guard_w * thr
                break
    for a in sc.allies:
        if "vip" in a.tags and sc.dist2(a, f) <= 4:       # attackers of the VIP first
            v += (3.0 - 0.4 * sc.dist2(a, f)) * (thr + 2.0) * 0.6
            break
    if prof.isolated_w:
        n = 0
        for o in sc.standing:
            if o is not f and sc.dist2(o, f) <= 2:
                n += 1
        v += prof.isolated_w * thr * (1.0 - min(2, n) / 2.0) * 0.6
    if inf.conds:
        cv = 0.0
        for cnd in inf.conds:
            cv = max(cv, COND_VALUE.get(cnd, 0.3))
        v += ph * cv * thr * 0.7 * (prof.prone_w if int(inf.conds[0]) == 10 else 1.0)
    return v


# ---------------------------------------------------------------------------------------------------- survey
def survey(sc, prof) -> Survey:
    sv = Survey()
    mask = sc.mask()
    post = []
    for k, ab, inf in sc.relevant():
        if not mask[k]:
            continue
        role = inf.role
        if role == "sattack":
            post.append((k, ab, inf))
            continue
        _HANDLERS[role](sv, sc, prof, k, ab, inf)
    for (k, ab, inf) in post:
        _h_sattack(sv, sc, prof, k, ab, inf)
    if sv.pot:
        sv.has_attack_now = any(c.tag.startswith("atk") for c in sv.cands)
    sc.flush()
    return sv


def _record_pot(sv, f, inf, v: float) -> None:
    p = sv.pot.get(f.eid)
    if p is None:
        p = sv.pot[f.eid] = Pot(f)
    if inf.melee:
        if v > p.mval:
            p.mval = v
        p.reach = max(p.reach, inf.reach)
    else:
        if v > p.rval:
            p.rval = v
        p.rng = max(p.rng, inf.rng)
        p.rngl = max(p.rngl, inf.rng_long)
    if v > sv.best_attack:
        sv.best_attack = v


# ---------------------------------------------------------------------------------------------------- attack
def _h_attack(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    eid = sc.eid
    cost = slot_cost(prof, inf)
    targets = sc.vis
    maxr = inf.reach if inf.melee else inf.rng_long
    pen = 1.0
    if inf.thrown:
        pen = 0.7
    if inf.offhand:
        pen = 0.95
    man = inf.kw.get("maneuver")
    if man in ("grapple", "shove_prone", "shove_push"):
        return        # maneuvers are scored by the tactical layer (prone / push), not as damage
    if man == "unarmed_strike":
        if sc.has_real_attack(k):
            return    # clearly dominated by a real weapon / natural attack
        pen = 0.9
    far = 0
    for f in sc.vis_sorted():
        d = sc.dist(f)
        if d > maxr + 1:
            far += 1
            if far > 3:
                break               # only the nearest few out-of-range foes matter for movement planning
        ph, ed = sc.est(k, f) if d <= maxr + 1 else sc.est_far(k, f)
        if ed <= 0 and not inf.conds:
            continue
        if not ctype_ok(inf, f):
            continue
        key = (f.eid, inf.melee)
        bed = sv.best_ed.get(key, 0.0)
        if ed < 0.85 * bed and not inf.conds and not inf.slot:
            continue                    # clearly dominated by another weapon / attack of the same kind on this foe
        if ed > bed:
            sv.best_ed[key] = ed
        v = attack_value(sc, prof, inf, f, ph, ed, ab) * pen
        _record_pot(sv, f, inf, v - cost)
        if d <= maxr and ab.valid_target(w, c, f.eid):
            sv.cands.append(Cand(v - cost, ab_act(k, f.eid), f"atk:{ab.id}>{f.eid}"))
    if prof.finish_downed:
        for f in sc.downed_foes:
            d = w.distance(eid, f.eid)
            if d <= maxr and w.sees(eid, f.eid) and ab.valid_target(w, c, f.eid):
                ph, ed = sc.est(k, f)
                v = attack_value(sc, prof, inf, f, ph, max(ed, 1.0), ab) * pen
                sv.cands.append(Cand(v - cost, ab_act(k, f.eid), f"fin:{ab.id}>{f.eid}"))


def _h_sattack(sv, sc, prof, k, ab, inf) -> None:
    """Monster specials whose damage the engine cannot estimate: valued relative to the best ordinary attack."""
    c, w = sc.c, sc.w
    base = max(sv.best_attack, 3.0) * inf.kw.get("sattack", 0.9)
    best = None
    for f in sc.vis:
        if w.distance(sc.eid, f.eid) > max(inf.reach, inf.rng, 1):
            continue
        if not ab.valid_target(w, c, f.eid):
            continue
        thr = sc.threat(f)
        v = base * (1.0 + 0.1 * (thr > 6)) - slot_cost(prof, inf)
        if best is None or v > best[0]:
            best = (v, f)
    if best is not None:
        sv.cands.append(Cand(best[0], ab_act(k, best[1].eid), f"atk:{ab.id}>{best[1].eid}"))


# ---------------------------------------------------------------------------------------------------- area helpers
def _area_points(sc, ab, inf) -> list:
    """Aim points worth evaluating for an area ability."""
    c = sc.c
    area = ab.area
    if area is None:
        return []
    foes = sc.vis
    if inf.shape == AreaShape.EMANATION:
        return [(c.x, c.y)]
    pts = []
    seen = set()
    near = sorted(foes, key=lambda f: max(abs(f.x - c.x), abs(f.y - c.y)))[:8]
    for f in near:
        p = (f.x, f.y)
        if p not in seen:
            seen.add(p)
            pts.append(p)
    if area.origin != "self" and inf.shape not in (AreaShape.CONE, AreaShape.LINE) and len(near) >= 2:
        r2 = max(1, inf.area_sq) * 2
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                a, b = near[i], near[j]
                if max(abs(a.x - b.x), abs(a.y - b.y)) <= r2:
                    p = ((a.x + b.x) // 2, (a.y + b.y) // 2)
                    if p not in seen:
                        seen.add(p)
                        pts.append(p)
            if len(pts) >= 14:
                break
    return pts


def _creatures_in(sc, cells) -> list:
    cm = sc.cellmap()
    out = {}
    for p in cells:
        o = cm.get((int(p[0]), int(p[1])))
        if o is not None:
            out[o.eid] = o
    return list(out.values())


def _point_legal(sc, ab, pt) -> bool:
    w, c = sc.w, sc.c
    if ab.target == TargetType.POINT:
        return bool(ab.valid_point(w, c, pt[0], pt[1]))
    if ab.target == TargetType.NONE:
        return w.map.in_bounds(pt[0], pt[1])
    return False


def _h_aoe(sv, sc, prof, k, ab, inf) -> None:
    from emberdeep.rules.spells import area_for
    c, w = sc.c, sc.w
    if not sc.vis:
        return
    cost = slot_cost(prof, inf)
    cp = conc_penalty(sc, prof, inf, 0.0)
    zone_mult = 1.8 if inf.zone else 1.0
    best = None
    for pt in _area_points(sc, ab, inf):
        if not _point_legal(sc, ab, pt):
            continue
        try:
            cells, _, _ = area_for(w, c, ab.area, pt)
        except Exception:  # noqa: BLE001
            continue
        hit = _creatures_in(sc, cells)
        if not hit:
            continue
        gain = loss = 0.0
        nfoe = 0
        for o in hit:
            if o.eid == sc.eid:
                _, ed = sc.est(k, o)
                loss += 1.5 * ed
                continue
            r = sc.rel(o)
            ph, ed = sc.est(k, o)
            if r == 0:
                if o.status & (S.DYING | S.STABLE | S.SURRENDERED):
                    gain += 0.1 * ed
                    continue
                nfoe += 1
                gain += attack_value(sc, prof, inf, o, ph, ed) * zone_mult
            elif r == 3:
                loss += prof.friendly_fire_w * (min(ed, o.hp + 0.0) + (4.0 if ed >= o.hp else 0.0))
                if inf.conds:
                    loss += ph * 2.0
            elif r == 2:
                loss += 8.0 + ed
            else:
                loss += 3.0 + ed
        if nfoe == 0:
            continue
        v = (gain * prof.aoe_w) - loss - cost - cp
        if best is None or v > best[0]:
            best = (v, pt, nfoe)
        # every foe also contributes to the "potential" table (range only matters for movement)
    if best is not None:
        v, pt, nfoe = best
        tgt = -1
        sv.cands.append(Cand(v, ab_act(k, tgt, pt), f"aoe:{ab.id}@{pt}x{nfoe}"))
        sv.best_attack = max(sv.best_attack, v)


# ---------------------------------------------------------------------------------------------------- control
_SLEEP_IDS = ("sleep",)


def _control_value(sc, prof, inf, k, f, slot: int = 1) -> float:
    ph, ed = sc.est(k, f)
    cv = 0.0
    for cnd in inf.conds:
        cv = max(cv, COND_VALUE.get(cnd, 0.3))
    if cv == 0.0:
        cv = 0.6
    if f.cond_mask & 0x30A0:        # already paralysed / stunned / unconscious / incapacitated
        return 0.0
    thr = sc.threat(f)
    pf = ph
    if f.hp <= 0:
        return 0.0
    return pf * cv * (thr * 1.4 + 1.5) * prof.control_w


def _h_control(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    if not sc.vis:
        return
    cost = slot_cost(prof, inf)
    cp = conc_penalty(sc, prof, inf, 0.0)
    if ab.area is not None:
        _control_area(sv, sc, prof, k, ab, inf, cost, cp)
        return
    for f in sc.vis:
        if not ctype_ok(inf, f):
            continue
        if ab.target in (TargetType.ENEMY, TargetType.CREATURE) and w.distance(sc.eid, f.eid) > max(inf.rng_long, inf.reach):
            continue
        if not ab.valid_target(w, c, f.eid):
            continue
        v = _control_value(sc, prof, inf, k, f) - cost - cp
        if inf.melee:
            v *= 0.9
        if v > 0.3:
            sv.cands.append(Cand(v, ab_act(k, f.eid), f"ctl:{ab.id}>{f.eid}"))


def _control_area(sv, sc, prof, k, ab, inf, cost, cp) -> None:
    from emberdeep.rules.spells import area_for
    c, w = sc.c, sc.w
    best = None
    sleepy = inf.sid in _SLEEP_IDS
    for pt in _area_points(sc, ab, inf):
        if not _point_legal(sc, ab, pt):
            continue
        try:
            cells, _, _ = area_for(w, c, ab.area, pt)
        except Exception:  # noqa: BLE001
            continue
        hit = _creatures_in(sc, cells)
        gain = loss = 0.0
        n = 0
        if sleepy:
            pool = 22.0
            hit_foes = sorted((o for o in hit if sc.rel(o) == 0 and o.ctype != 13 and not o.cond_mask & 0x2000),
                              key=lambda o: o.hp)
            for o in hit_foes:
                if o.hp > pool:
                    break
                pool -= o.hp
                n += 1
                gain += 1.5 * (sc.threat(o) * 1.4 + 1.5) * prof.control_w
            for o in hit:
                if sc.rel(o) == 3 and o.ctype != 13:
                    loss += 8.0
        else:
            for o in hit:
                if o.eid == sc.eid:
                    loss += 5.0
                    continue
                r = sc.rel(o)
                if r == 0:
                    if o.status & (S.DYING | S.STABLE | S.SURRENDERED):
                        continue
                    if not ctype_ok(inf, o):
                        continue
                    v = _control_value(sc, prof, inf, k, o)
                    if v > 0:
                        n += 1
                        gain += v
                elif r == 3:
                    ph, _ = sc.est(k, o)
                    loss += ph * 5.0 * prof.friendly_fire_w
                elif r == 2:
                    loss += 10.0
                else:
                    loss += 3.0
        if n == 0:
            continue
        v = gain - loss - cost - cp
        if best is None or v > best[0]:
            best = (v, pt, n)
    if best is not None and best[0] > 0.3:
        v, pt, n = best
        sv.cands.append(Cand(v, ab_act(k, -1, pt), f"ctl:{ab.id}@{pt}x{n}"))


def _h_debuff(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    if not sc.vis:
        return
    cost = slot_cost(prof, inf)
    cp = conc_penalty(sc, prof, inf, 0.0)
    if ab.area is not None:
        _control_area(sv, sc, prof, k, ab, inf, cost, cp)
        return
    for f in sc.vis:
        if not ctype_ok(inf, f) or not ab.valid_target(w, c, f.eid):
            continue
        bid = inf.buff_id or inf.sid
        if bid and any(e.ID == bid and e.source == sc.eid for e in f.effects):
            continue
        ph, ed = sc.est(k, f)
        thr = sc.threat(f)
        v = ph * (0.5 * thr + 1.0) * prof.control_w - cost - cp
        if ed > 0:
            v += attack_value(sc, prof, inf, f, ph, ed)
        if v > 0.3:
            sv.cands.append(Cand(v, ab_act(k, f.eid), f"dbf:{ab.id}>{f.eid}"))


# ---------------------------------------------------------------------------------------------------- healing
def _potion_heal(c) -> float:
    it = best_potion(c)
    if it is None:
        return 0.0
    try:
        return D.average(it.def_.heal)
    except Exception:  # noqa: BLE001
        return 5.0


def _feature_heal_amount(sc, ab, inf) -> float:
    c = sc.c
    if "second_wind" in ab.id:
        return 5.5 + c.class_levels.get("fighter", c.level)
    res = inf.resource
    pool = c.resources.get(res, 0) if res else 0
    if "lay_on_hands" in ab.id:
        return float(min(pool, 12))
    return 6.0


def _h_heal(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    eid = sc.eid
    if ab.kind == "basic":
        amount = _potion_heal(c)
    elif ab.kind == "feature":
        amount = _feature_heal_amount(sc, ab, inf)
    else:
        amount = heal_amount(inf, c)
    if amount <= 0:
        return
    cost = slot_cost(prof, inf)
    cands = list(sc.allies) + [c] + list(sc.downed_allies)
    if ab.id == "drink_potion":
        cands = [c]
    elif ab.id == "administer_potion":
        cands = list(sc.downed_allies) + [a for a in sc.allies if a.hp < a.max_hp * 0.4]
    for a in cands:
        dn = bool(a.status & (S.DYING | S.STABLE))
        if a.status & S.DEAD:
            continue
        if not dn and a.hp >= a.max_hp:
            continue
        hpf = a.hp / max(1, a.max_hp)
        if dn:
            v = max(amount, 6.0) * 3.0 * (1.0 if a.status & S.DYING else 0.65)
        else:
            missing = a.max_hp - a.hp
            eff = min(amount, missing)
            if hpf > prof.heal_thr and eff < 0.75 * amount:
                continue
            urg = 1.0 + 2.5 * max(0.0, 0.55 - hpf) + (0.8 if a is c and hpf < 0.35 and sc.vis else 0.0)
            v = eff * urg
        v = v * prof.heal_w - cost
        if a is c and inf.tgt in (TargetType.ALLY,):
            continue
        # position potential for the movement planner
        sv.heal_pot.append((a, v, max(inf.rng, inf.reach, 1)))
        if v <= 0.3:
            continue
        if ab.target in (TargetType.SELF, TargetType.NONE):
            if a is not c:
                continue
            sv.cands.append(Cand(v, ab_act(k, eid), f"heal:{ab.id}>{a.eid}"))
            continue
        if w.distance(eid, a.eid) > max(inf.rng, inf.reach, 1) and a is not c:
            continue
        if ab.valid_target(w, c, a.eid):
            sv.cands.append(Cand(v, ab_act(k, a.eid), f"heal:{ab.id}>{a.eid}"))


def _h_stabilize(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    for a in sc.downed_allies:
        if not a.status & S.DYING:
            continue
        v = 5.0 * prof.heal_w
        if w.distance(sc.eid, a.eid) <= max(1, inf.rng) and ab.valid_target(w, c, a.eid):
            sv.cands.append(Cand(v, ab_act(k, a.eid), f"stab:{ab.id}>{a.eid}"))
            sv.heal_pot.append((a, v, 1))


# ---------------------------------------------------------------------------------------------------- buffs
def _fight_factor(sc) -> float:
    if not sc.standing:
        return 0.15
    c = sc.c
    near = sc.vis
    if not near:
        return 0.35
    d = min(sc.w.distance(sc.eid, f.eid) for f in near)
    return 1.0 if d <= 14 else 0.6


def _h_buff(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    bid = inf.buff_id or inf.sid or ab.id
    base = BUFF_VALUE.get(bid, 2.0)
    if base <= 0.0 or prof.buff_w <= 0:
        return
    ff = _fight_factor(sc)
    cost = slot_cost(prof, inf)
    cp = conc_penalty(sc, prof, inf, 0.0)
    pool = [c] if inf.self_only or ab.target in (TargetType.SELF, TargetType.NONE) else \
        sorted([c] + list(sc.allies), key=lambda a: -_ally_value(sc, a))
    recipients = []
    for a in pool:
        if bid and any(e.ID == bid for e in a.effects):
            continue
        if bid == "mage_armor" and a.equipped.get("armor") is not None:
            continue
        if inf.self_only and a is not c:
            continue
        if a is c and ab.target in (TargetType.ALLY,):
            continue
        if a is not c and w.distance(sc.eid, a.eid) > max(inf.rng, inf.reach, 1):
            continue
        recipients.append(a)
    if not recipients:
        return
    n = min(max(1, inf.targets_max), len(recipients))
    v = base * n * prof.buff_w * ff - cost - cp
    if v <= 0.4:
        return
    for a in recipients:
        if ab.target in (TargetType.SELF, TargetType.NONE):
            if a is not c:
                continue
            sv.cands.append(Cand(v, ab_act(k, sc.eid), f"buff:{ab.id}"))
            return
        if ab.valid_target(w, c, a.eid):
            sv.cands.append(Cand(v, ab_act(k, a.eid), f"buff:{ab.id}>{a.eid}"))
            return


def _ally_value(sc, a) -> float:
    o = sc.offense(a) if a is not sc.c else {"m": 5, "r": 5}
    return max(o["m"], o["r"]) + 0.01 * a.max_hp


def _h_mark(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    if c.status & S.CONCENTRATING and inf.conc:
        return
    cost = slot_cost(prof, inf)
    best = None
    for f in sc.vis:
        if any(e.ID == (inf.buff_id or inf.sid) and e.source == sc.eid for e in f.effects):
            continue
        if not ab.valid_target(w, c, f.eid):
            continue
        v = 3.0 + 0.3 * sc.threat(f) - cost
        if best is None or v > best[0]:
            best = (v, f)
    if best is not None and best[0] > 0.5:
        sv.cands.append(Cand(best[0], ab_act(k, best[1].eid), f"mark:{ab.id}>{best[1].eid}"))


def _h_summon(sv, sc, prof, k, ab, inf) -> None:
    c, w = sc.c, sc.w
    if not sc.vis:
        return
    mine = sum(1 for o in w.creatures if o.present and o.summoner == sc.eid)
    base = inf.kw.get("summon_value", 8.0)
    if mine:
        base *= 0.25
    cp = conc_penalty(sc, prof, inf, 0.0)
    v = base - slot_cost(prof, inf) - cp
    if v <= 0.5:
        return
    if ab.target in (TargetType.POINT, TargetType.NONE):
        best = min(sc.vis, key=lambda f: w.distance(sc.eid, f.eid))
        pt = (best.x, best.y)
        if ab.target == TargetType.POINT and not ab.valid_point(w, c, pt[0], pt[1]):
            return
        sv.cands.append(Cand(v, ab_act(k, -1, pt), f"summon:{ab.id}"))
    else:
        for f in sc.vis:
            if ab.valid_target(w, c, f.eid):
                sv.cands.append(Cand(v, ab_act(k, f.eid), f"summon:{ab.id}"))
                return


# ---------------------------------------------------------------------------------------------------- stances
def _h_stance(sv, sc, prof, k, ab, inf) -> None:
    """Rage / action surge / flurry / bardic inspiration / turn undead ... (all optional, none assumed to exist)."""
    c, w = sc.c, sc.w
    st = inf.stance
    ff = prof.stance_w
    foes = sc.vis
    if not foes and st not in ("inspire",):
        return
    near = min((w.distance(sc.eid, f.eid) for f in foes), default=99)
    ts = c.ts
    v = 0.0
    tgt = -1
    if st == "rage":
        if c.status & S.RAGING:
            return
        reach = max(1, 1 + c.ts.movement_left // 5)
        if near <= reach + 1:
            v = 7.0
    elif st == "reckless":
        if near <= 1 and not c.status & S.DEAD:
            v = 2.5
    elif st == "surge":
        if ts.action_used and ts.attacks_left == 0 and not ts.action_surge_used:
            v = max(4.0, mem(c).get("last_atk", 6.0)) * 0.9
            if not any(w.distance(sc.eid, f.eid) <= 6 for f in foes):
                v = 0.0
    elif st == "flurry":
        if ts.attack_committed or ts.action_used:
            if near <= 1:
                v = 4.0
                tgt = _nearest_target(sc, ab, k)
                if tgt < 0 and ab.target not in (TargetType.SELF, TargetType.NONE):
                    return
    elif st == "defense":
        if near <= 1 and sc.hpf < 0.5:
            v = 3.0
    elif st == "inspire":
        allies = [a for a in sc.allies if a is not c and w.distance(sc.eid, a.eid) <= max(2, inf.rng)]
        if allies and foes:
            v = 2.5
            a = max(allies, key=lambda a: _ally_value(sc, a))
            tgt = a.eid
    elif st == "turn":
        und = [f for f in foes if f.ctype == 13 and w.distance(sc.eid, f.eid) <= 6]
        if und:
            v = 4.0 * len(und)
    elif st == "channel":
        und = [f for f in foes if f.ctype in (13, 7) and w.distance(sc.eid, f.eid) <= 6]
        if und:
            v = 4.0 * len(und)
    if v * ff <= 0.5:
        return
    if tgt < 0:
        if ab.target in (TargetType.SELF, TargetType.NONE):
            tgt = sc.eid
        elif ab.target == TargetType.POINT:
            return
        else:
            tgt = _nearest_target(sc, ab, k)
            if tgt < 0:
                return
    if not _target_legal(sc, ab, tgt):
        return
    sv.cands.append(Cand(v * ff, ab_act(k, tgt), f"stance:{ab.id}"))


def _nearest_target(sc, ab, k) -> int:
    w, c = sc.w, sc.c
    best, bd = -1, 99
    for f in sc.vis:
        d = w.distance(sc.eid, f.eid)
        if d < bd and ab.valid_target(w, c, f.eid):
            best, bd = f.eid, d
    return best


def _target_legal(sc, ab, tgt) -> bool:
    if ab.target in (TargetType.SELF, TargetType.NONE):
        return True
    return bool(ab.valid_target(sc.w, sc.c, tgt))


_HANDLERS = {
    "attack": _h_attack, "aoe": _h_aoe, "control": _h_control, "debuff": _h_debuff, "heal": _h_heal,
    "stabilize": _h_stabilize, "buff": _h_buff, "mark": _h_mark, "summon": _h_summon, "stance": _h_stance,
}
