# data-spells contract requests

1. `SpellDef.temp_hp_per_slot_above: str = ""` — False Life gives +5 temp HP per slot above 1st; currently handled by hook `false_life`.
2. `SpellDef.pool_dice` / generic HP-pool support (Sleep 5d8 +2d8, Color Spray 6d10 +2d10) — currently carried in effect params `pool`/`pool_per_slot` and handled by hooks.
3. `ConditionSpec` has no "end on shake/awaken" flag (Sleep, Hypnotic Pattern); handled by hooks.
