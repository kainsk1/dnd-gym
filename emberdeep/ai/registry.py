"""Policy registry and per-creature policy resolution.

``get_policy(name, seed=None)``   build / fetch a policy by name
``policy_for(world, eid)``        the policy that should act for a creature (ai override > tags > monster profile > class)
``scripted_decider(world, dec)``  plug into ``world.auto_decider`` to answer reactions with the right policy
"""
from __future__ import annotations

from emberdeep.ai.base import Policy
from emberdeep.ai.profiles import PROFILES

_CACHE: dict[str, Policy] = {}

#: names understood by :func:`get_policy` (hero_<class> for every SRD class)
CLASS_NAMES = ("barbarian", "bard", "cleric", "druid", "fighter", "monk", "paladin", "ranger", "rogue", "sorcerer",
               "warlock", "wizard")
POLICY_NAMES = ("random", "idle") + tuple(n for n in PROFILES if n != "idle") + ("hero",) + tuple(
    f"hero_{c}" for c in CLASS_NAMES)


def get_policy(name: str, seed: int | None = None) -> Policy:
    """Policy by registry name.  Policies built without a ``seed`` are stateless and cached (shared); with a ``seed``
    a fresh instance is returned (only ``random`` actually uses it)."""
    key = (name or "").strip().lower()
    if seed is None and key in _CACHE:
        return _CACHE[key]
    pol = _build(key, seed)
    if seed is None:
        _CACHE[key] = pol
    return pol


def _build(key: str, seed: int | None) -> Policy:
    if key == "random":
        from emberdeep.ai.random_policy import RandomPolicy
        return RandomPolicy(seed)
    if key in PROFILES:
        return _make_profile_policy(key, seed)
    if key == "hero" or key.startswith("hero_"):
        from emberdeep.ai.heroes import HeroPolicy
        cls = key[5:] if key.startswith("hero_") else ""
        if cls and cls not in CLASS_NAMES:
            raise KeyError(f"unknown policy '{key}'")
        return HeroPolicy(cls, seed=seed)
    raise KeyError(f"unknown policy '{key}' (known: {', '.join(POLICY_NAMES)})")


def _make_profile_policy(key: str, seed: int | None) -> Policy:
    from emberdeep.ai.policy import UtilityPolicy
    if key == "vip":
        from emberdeep.ai.vip import VipPolicy
        return VipPolicy(seed=seed)
    return UtilityPolicy(PROFILES[key], seed=seed)


def _pc_class(c) -> str:
    if c.class_levels:
        return max(c.class_levels.items(), key=lambda kv: kv[1])[0]
    return c.template_id or ""


def policy_name_for(world, eid: int) -> str:
    c = world.entities[eid]
    tags = c.tags or ()
    name = (c.ai or "").strip().lower()
    if "vip" in tags:
        return "vip"
    if name and name in POLICY_NAMES:
        return name
    if c.is_pc:
        cls = _pc_class(c)
        return f"hero_{cls}" if cls in CLASS_NAMES else "hero"
    if c.summoner >= 0:
        return "summon"
    if "boss" in tags:
        return "boss"
    return "brute"


def policy_for(world, eid: int) -> Policy:
    """The scripted policy for creature ``eid`` (see module docstring for precedence)."""
    return get_policy(policy_name_for(world, eid))


def scripted_decider(world, decision) -> int:
    """``world.auto_decider`` implementation: route a pending decision to its creature's policy."""
    return int(policy_for(world, decision.eid).decide(world, decision))
