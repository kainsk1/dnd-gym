"""Scripted AI for Emberdeep: utility-based policies, behaviour profiles, hero bots, morale / social behaviour.

Public API (see ``docs/AI.md``)::

    from emberdeep.ai import get_policy, policy_for, scripted_decider, Policy
    world.auto_decider = scripted_decider
    act = policy_for(world, eid).act(world, eid)
    world.step(act)
"""
from emberdeep.ai.base import Policy
from emberdeep.ai.registry import POLICY_NAMES, get_policy, policy_for, policy_name_for, scripted_decider

__all__ = ["Policy", "get_policy", "policy_for", "policy_name_for", "scripted_decider", "POLICY_NAMES"]
