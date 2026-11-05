"""Monster traits, part 3: movement-triggered attacks (charge, pounce, trampling charge) and rampage."""
from __future__ import annotations

from emberdeep.core import resolution as R
from emberdeep.core.abilities import FeatureAbility
from emberdeep.core.types import ActionCost, AttackKind, Condition, TargetType
from emberdeep.data.schema import AttackDef
from emberdeep.rules.content import monster_util as U
from emberdeep.rules.registry import Trait, register_trait


class _Runner(Trait):
    """Records the unit steps walked during the creature's own turn."""
    HOOKS = frozenset({"on_move", "on_turn_start"})

    def on_turn_start(self, world, me):
        self.state["trail"] = ()
        self.state["hit"] = None

    def on_move(self, world, me, frm, to):
        d = (to[0] - frm[0], to[1] - frm[1])
        if d != (0, 0):
            self.state["trail"] = tuple(self.state.get("trail", ())) + (d,)

    def straight_ft(self, world, me, target) -> int:
        """Feet walked in a straight line (the last run of identical steps) that points toward ``target``."""
        tr = self.state.get("trail", ())
        if not tr or world.active != me.eid:
            return 0
        d = tr[-1]
        n = 0
        for s in reversed(tr):
            if s != d:
                break
            n += 1
        sx, sy = me.x - n * d[0], me.y - n * d[1]
        tx = target.x + (target.size_sq - 1) / 2.0
        ty = target.y + (target.size_sq - 1) / 2.0
        if (tx - sx) * d[0] + (ty - sy) * d[1] <= 0:
            return 0
        return 5 * n

    def attack_name(self, me) -> str:
        """The attack the trait modifies: ``params['attack']`` or the first melee attack in the stat block."""
        a = self.params.get("attack")
        if a:
            return a
        m = U.mdef(me)
        for x in m.actions:
            if isinstance(x, AttackDef) and x.kind == AttackKind.MELEE_WEAPON:
                return x.id
        return ""


def _knock_prone(world, src, target, params) -> None:
    eff = str(params.get("effect", "prone"))
    if eff.startswith("push"):
        dx = (target.x > src.x) - (target.x < src.x)
        dy = (target.y > src.y) - (target.y < src.y)
        world.forced_move(target, dx, dy, 2, src.eid)
    R.add_condition(world, target, Condition.PRONE, src.eid, -1)


@register_trait("charge")
class Charge(_Runner):
    """Moved at least ``min_move_ft`` straight toward a target and then hit it on the same turn: extra ``damage`` and, if
    ``save`` is given, a saving throw (DC ``dc``) or the ``effect`` (prone, or pushed 10 ft and prone)."""
    HOOKS = frozenset({"on_move", "on_turn_start", "on_hit"})

    def on_hit(self, world, ctx):
        a = ctx.attacker
        if a.eid != self.me or ctx.ranged or ctx.is_spell or ctx.ability_id != self.attack_name(a):
            return
        if self.straight_ft(world, a, ctx.target) < int(self.params.get("min_move_ft", 20)):
            return
        dmg = self.params.get("damage")
        if dmg:
            dice, dt = U.parse_dmg(dmg)
            ctx.add_damage(dice, dt)
        U.log(world, f"{a.name} charges {ctx.target.name}", a.eid)
        if self.params.get("save"):
            t = ctx.target
            if not R.saving_throw(world, t, U.ability_of(self.params["save"]), int(self.params.get("dc", 13)), a.eid,
                                  name="charge", tags=("prone", "knock_prone")):
                _knock_prone(world, a, t, self.params)


@register_trait("pounce")
class Pounce(_Runner):
    """Moved at least ``min_move_ft`` straight toward a creature and then hit it with ``attack``: STR save or prone; against a
    prone target it may make ``bonus_attack`` as a bonus action (granted as an ability for the rest of the turn)."""
    HOOKS = frozenset({"on_move", "on_turn_start", "on_hit", "grant_abilities"})

    def on_hit(self, world, ctx):
        a, t = ctx.attacker, ctx.target
        if a.eid != self.me or ctx.ranged or ctx.is_spell or ctx.ability_id != self.attack_name(a):
            return
        if self.straight_ft(world, a, t) < int(self.params.get("min_move_ft", 20)):
            return
        U.log(world, f"{a.name} pounces on {t.name}", a.eid)
        if not R.saving_throw(world, t, U.ability_of(self.params.get("save", "str")), int(self.params.get("dc", 12)), a.eid,
                              name=self.ID, tags=("prone", "knock_prone")):
            _knock_prone(world, a, t, self.params)
        self.state["hit"] = t.eid

    def grant_abilities(self, world, me):
        bonus = self.params.get("bonus_attack")
        if not bonus:
            return []
        return [FeatureAbility(f"{self.ID}_{bonus}", f"{self.ID.replace('_', ' ').title()}: {bonus}", "feature",
                               ActionCost.BONUS_ACTION, TargetType.CREATURE, 1, 0, 1, None, -1, None, "", "bite",
                               ("attack", "melee"), {"hook": self})]

    def _victim(self, world, me):
        eid = self.state.get("hit")
        if eid is None or world.active != me.eid:
            return None
        t = world.entities[eid]
        if not t.present or not t.cond_mask & (1 << int(Condition.PRONE)) or world.distance(me.eid, eid) > 1:
            return None
        return t

    def can_use(self, world, actor, ab):
        return self._victim(world, actor) is not None

    def valid_target(self, world, actor, ab, eid):
        v = self._victim(world, actor)
        return v is not None and v.eid == eid

    def use(self, world, actor, ab, eid, point):
        t = world.entities[eid]
        atk = U.find_ability(world, actor, self.params.get("bonus_attack"))
        if atk is not None:
            world.free_attack(actor, atk, t)


register_trait("trampling_charge")(type("TramplingCharge", (Pounce,), {
    "__doc__": "Warhorse: like Pounce with the hooves attack (STR DC 14 or prone, then another hooves attack as a bonus "
               "action against a prone target)."}))


@register_trait("rampage")
class Rampage(Trait):
    """When it reduces a creature to 0 HP with a melee attack on its turn, the 'rampage' bonus action becomes available
    (move up to half its speed and bite)."""
    HOOKS = frozenset({"on_hit", "on_turn_start"})

    def on_turn_start(self, world, me):
        U.mh(me)["rampage_ready"] = -1

    def on_hit(self, world, ctx):
        a = ctx.attacker
        if a.eid != self.me or ctx.ranged or ctx.is_spell or world.active != a.eid:
            return
        tid = ctx.target.eid
        me = self.me

        def check(w, me=me, tid=tid):
            c = w.entities[me]
            t = w.entities[tid]
            if t.hp <= 0 and c.present:
                U.mh(c)["rampage_ready"] = tid

        world.defer(check)
