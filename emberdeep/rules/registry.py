"""Hook registries and base classes (DESIGN §5).

Every bespoke piece of rules logic is a small class registered under a string id with a decorator::

    @register_feature("sneak_attack")
    class SneakAttack(Feature):
        HOOKS = frozenset({"damage_mods"})
        def damage_mods(self, world, ctx): ...

The engine only calls methods named in ``HOOKS`` (resolved once per creature and cached), so the
non-implemented hooks cost nothing.

Hook objects are instantiated *per owner* (features/traits: once per creature; effects: once per
application).  Instances carry ``me`` (owner eid), ``params`` (data parameters, shared/read-only) and
``state`` (a small dict of per-instance mutable state, copied on clone).

Hook methods (all optional, declare them in ``HOOKS``)::

    on_turn_start(world, me)           on_turn_end(world, me)
    attack_mods(world, ctx)            defense_mods(world, ctx)
    damage_mods(world, ctx)            incoming_damage_mods(world, ctx)
    save_mods(world, ctx)              check_mods(world, ctx)
    on_hit(world, ctx)                 on_damaged(world, me, amount, src)
    on_reduced_to_zero(world, me, ctx) -> bool        on_kill(world, me, victim)
    on_move(world, me, frm, to)        ac_bonus(world, me) -> int
    speed_bonus(world, me) -> int      grant_abilities(world, me) -> list[Ability]
    ac_base(world, me) -> int | None   (alternative base AC formula, e.g. Unarmored Defense; highest wins)
    attacks_per_action(world, me) -> int   (Extra Attack: returns the number of attacks; max wins)
    reaction_options(world, me, kind, ctx) -> list[DecisionOption]
    apply_reaction(world, me, kind, ctx, option)       (resolve a chosen option)
    on_short_rest(world, me)           on_long_rest(world, me)
    on_cast(world, me, spell, slot)    (after the owner casts a spell)
    on_round_start(world, me)
    on_condition(world, me, cond, adding) -> bool      (return True to *prevent* adding the condition)
    modify_speed(world, me, speed) -> int              (multiplicative / replacing effects)
    resist_mask(world, me, dtype, magical) -> int      (+1 resist, -1 vulnerable, 2 immune)
    on_attack_declared(world, ctx)                     (right before the attack roll)
    unarmed_damage(world, me) -> str | None            (dice for unarmed strikes, e.g. monk martial arts)
    can_hide(world, me, observer) -> bool              (traits like Naturally Stealthy)
    initiative_bonus(world, me) -> int
"""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

ALL_HOOKS = (
    "on_turn_start", "on_turn_end", "attack_mods", "defense_mods", "damage_mods", "incoming_damage_mods",
    "save_mods", "check_mods", "on_hit", "on_damaged", "on_reduced_to_zero", "on_kill", "on_move",
    "ac_bonus", "speed_bonus", "grant_abilities", "ac_base", "attacks_per_action", "reaction_options",
    "apply_reaction", "on_short_rest", "on_long_rest", "on_cast", "on_round_start", "on_condition",
    "modify_speed", "resist_mask", "on_attack_declared", "unarmed_damage", "can_hide", "initiative_bonus", "on_spawn", "force_save_mods", "after_d20",
    "observe_d20", "ally_save_mods", "modify_healing", "death_save_mods", "veto_exhaustion", "avoids_opportunity",
    "can_pass_through", "may_share_space", "ignores_difficult", "weapon_ability", "ability_blocked", "target_filter",
    "economy_mods", "spell_range_mult", "ignore_components", "before_cast", "on_fall", "projectile_block",
)


class HookBase:
    """Base class of all hook implementations."""
    HOOKS: frozenset = frozenset()
    ID: str = ""

    def __init__(self, me: int = -1, params: dict | None = None):
        self.me = me
        self.params = params if params is not None else {}
        self.state: dict[str, Any] = {}

    def clone(self):
        """Shallow-structural copy used by World.clone (state dict copied)."""
        n = object.__new__(type(self))
        n.__dict__.update(self.__dict__)
        n.state = dict(self.state)
        return n

    @property
    def id(self) -> str:
        return self.ID


class Feature(HookBase):
    """Class feature / subclass feature / feat implementation."""
    fdef = None   # FeatureDef (set by the factory)


class Trait(HookBase):
    """Racial or monster trait implementation."""


class Registry:
    """A name -> class mapping with a decorator."""

    def __init__(self, name: str):
        self.name = name
        self._d: dict[str, type] = {}

    def register(self, id_: str) -> Callable[[type], type]:
        def deco(cls):
            if id_ in self._d and self._d[id_] is not cls:
                log.debug("%s registry: overriding %s", self.name, id_)
            cls.ID = id_
            self._d[id_] = cls
            return cls
        return deco

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def __getitem__(self, k: str):
        return self._d[k]

    def get(self, k: str, default=None):
        return self._d.get(k, default)

    def ids(self) -> list[str]:
        return sorted(self._d)

    def items(self):
        return self._d.items()

    def __len__(self):
        return len(self._d)


FEATURES = Registry("features")
TRAITS = Registry("traits")
EFFECTS = Registry("effects")
ZONES = Registry("zones")
SPELL_HOOKS = Registry("spell_hooks")
ITEM_HOOKS = Registry("item_hooks")
OBJECT_HOOKS = Registry("object_hooks")
SPECIAL_ACTIONS = Registry("special_actions")   # SpecialActionDef.hook ids

register_feature = FEATURES.register
register_trait = TRAITS.register
register_effect = EFFECTS.register
register_zone = ZONES.register
register_spell_hook = SPELL_HOOKS.register
register_item_hook = ITEM_HOOKS.register
register_object_hook = OBJECT_HOOKS.register
register_special_action = SPECIAL_ACTIONS.register

_warned: set[str] = set()


def warn_once(msg: str) -> None:
    """Log a debug message only once per process."""
    if msg not in _warned:
        _warned.add(msg)
        log.debug(msg)


def hooks_of(cls: type) -> frozenset:
    """Implemented hook names of ``cls`` (explicit HOOKS plus auto-detected overrides of ALL_HOOKS)."""
    h = set(getattr(cls, "HOOKS", ()))
    for n in ALL_HOOKS:
        if n in cls.__dict__ or any(n in b.__dict__ for b in cls.__mro__[:-1]):
            h.add(n)
    return frozenset(h)
