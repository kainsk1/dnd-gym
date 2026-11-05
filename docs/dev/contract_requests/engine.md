# Contract requests from the engine

Applied (append-only):

* `core/types.py`: `DecisionKind.LEGENDARY = 9` — a creature with legendary actions is asked (after another creature's turn
  in combat) which legendary action to take; scripted creatures use `world.auto_decider`.

Requests / notes for the orchestrator:

* `data/schema.py`: `SpellDef.temp_hp_per_slot_above` (announced by the orchestrator) is read with `getattr(..., "")`, so
  the engine works with and without it.
* `core/events.py`: no new members were needed. Event `data` keys that are not spelled out in the contract but are emitted:
  `ATTACK.data.adv_src / dis_src` (lists of advantage/disadvantage source tags), `SAVE.data.rolls / adv`,
  `ABILITY_USE.data.targets / area / point` for monster save actions, `ZONE_START.data.attached_to`.
