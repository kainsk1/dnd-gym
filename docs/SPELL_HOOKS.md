# Spell hooks, effects and zones (spell-hooks)

Code: `emberdeep/rules/content/spell_*.py`, `item_*.py` (auto-imported). Tests: `tests/test_spell_hooks.py`, `tests/test_spells_all.py`, `tests/test_items.py`.
Engine requests: `docs/dev/contract_requests/spell-hooks.md`. Caster options: `spell_common.set_choice(caster, spell_id, key)` or an agent decision.

Replaced registrations (later wins): zones `moonbeam fog_cloud stinking_cloud sleet_storm silence darkness daylight ice_storm guardian_of_faith gust_of_wind wind_wall flaming_sphere call_lightning`; effects `slow chill_touch hunters_mark(_tracker) aid heroism invisibility true_strike hold_person helped (Help wakes sleepers)`.
The old `spell_patches.py` shims are gone (engine hooks `modify_healing`, `death_save_mods`, `ON_FALL`). v1.1 additions: engine hooks `casting_time`, `on_targeted`, `economy_mods`, `before_cast`; spells with options (command, blindness_deafness, enhance_ability, enlarge_reduce, lesser_restoration, protection_from_energy, bestow_curse, calm_emotions, suggestion, conjure_animals, animate_dead) ask agent casters (`SPELL_CHOICE`) and use context-aware heuristics for scripted casters. Summons: `ai=='summon'`, own group initiative.

| Spell | Lvl | Implementation / deviations |
|---|---|---|
| acid_splash | 0 | generic; 2nd target within 5 ft of the first |
| chill_touch | 0 | hook (generic) + `chill_touch` effect: no healing, undead disadvantage vs caster, until end of caster's next turn |
| dancing_lights | 0 | hook: 4 light zones, `dancing_lights_move` bonus action (60 ft, <=120 ft from caster) |
| eldritch_blast | 0 | base hook; rays by level |
| fire_bolt | 0 | generic interpreter |
| guidance | 0 | generic `guidance_die` |
| light | 0 | hook: `light_source` effect on a creature (hostile DEX save) or light zone on an object. Cannot target self (engine target type) |
| mage_hand | 0 | hook + `mage_hand` effect: `mage_hand_use` opens/closes unlocked doors, opens unlocked chests (trap fires harmlessly), pulls levers, fetches <=10 lb of loot, range 30 ft |
| minor_illusion | 0 | hook + opaque `minor_illusion` zone; ends when walked into or when a hostile within range wins one free INT (Investigation) check vs DC at a round start (deviation: no action cost). Discernment is per observer: the creature that discerns it, or walks into it, sees through it; everyone else is still blocked (`World.discern_illusion`) |
| poison_spray | 0 | generic interpreter |
| produce_flame | 0 | hook + effect granting `produce_flame_throw` (ranged spell attack, ends flame); light 10/10 |
| ray_of_frost | 0 | generic interpreter |
| resistance | 0 | generic interpreter |
| sacred_flame | 0 | generic interpreter |
| shillelagh | 0 | hook (needs club/quarterstaff) + effect: spell mod, d8, magical |
| shocking_grasp | 0 | hook: advantage vs metal armor, `no_reactions` until start of its next turn |
| spare_the_dying | 0 | base hook |
| thaumaturgy | 0 | hook + effect: advantage on Intimidation |
| true_strike | 0 | effect moves itself onto the caster (data attaches it to the target) |
| vicious_mockery | 0 | effect: disadvantage on next attack until end of its next turn |
| animal_friendship | 1 | hook: beasts INT<=3, 24 h |
| bane | 1 | generic interpreter |
| bless | 1 | generic interpreter |
| burning_hands | 1 | generic interpreter |
| charm_person | 1 | hook: humanoids, advantage when in combat, extra targets chosen among humanoids; `charmed_friend` BEHAVIOR friendly_to; harm by caster side ends it. `charmer_edge` effect: the charmer has advantage on CHA checks against the charmed creature (parley passes the target) |
| color_spray | 1 | hook: 6d10(+2d10/slot) HP pool, ascending HP, blinded until end of caster's next turn |
| command | 1 | hook + `commanded`: executed at the target's turn start (approach/flee move it, drop drops weapons, grovel prone, halt); turn ends. Understanding = INT>=4, not undead |
| cure_wounds | 1 | generic interpreter |
| detect_magic | 1 | hook + effect: LOG with aura ids, `detect_magic_study` (school) |
| divine_favor | 1 | generic interpreter |
| entangle | 1 | generic interpreter |
| expeditious_retreat | 1 | generic interpreter |
| faerie_fire | 1 | generic interpreter |
| false_life | 1 | hook: 1d4+4 +5/slot above 1st |
| feather_fall | 1 | reaction hook (fall wrapper in spell_patches) + no-damage effect |
| fog_cloud | 1 | zone override: radius +20 ft per slot |
| goodberry | 1 | hook: 10 `goodberry` items (heal 1; potency loss after 24 h not modelled) |
| grease | 1 | generic interpreter |
| guiding_bolt | 1 | generic interpreter |
| healing_word | 1 | generic interpreter |
| hellish_rebuke | 1 | generic interpreter |
| heroism | 1 | effect (item_effects): temp HP = caster mod, or `temp_hp_per_turn` (potion) |
| hideous_laughter | 1 | effect: INT<5 immune, save with advantage on damage |
| hunters_mark | 1 | hook + tracker effect: +1d6, `hunters_mark_move`, duration by slot |
| inflict_wounds | 1 | generic interpreter |
| jump | 1 | marker `JUMP_MULT` |
| longstrider | 1 | generic; touch spells reach one creature (engine) |
| mage_armor | 1 | generic |
| magic_missile | 1 | generic |
| protection_from_evil_and_good | 1 | generic interpreter |
| sanctuary | 1 | hook + effect: attacker WIS save or the attack is lost; ends on offense. single-target harmful spells too (`on_targeted`: WIS save, lost on failure); area effects are not stopped |
| shield | 1 | base reaction hook |
| shield_of_faith | 1 | generic interpreter |
| sleep | 1 | hook: 5d8(+2d8/slot) pool, ascending HP, undead/charm-immune skipped; damage or Help (shake) wakes |
| thunderwave | 1 | generic interpreter |
| acid_arrow | 2 | hook + effect: 4d4 now, 2d4 at end of target's next turn; miss = half, no later damage |
| aid | 2 | hook + effect (revives from 0 HP): +5 (+5/slot above 2nd) |
| barkskin | 2 | generic |
| blindness_deafness | 2 | hook: option, CON save each turn end, +1 target/slot |
| blur | 2 | generic |
| calm_emotions | 2 | hook + effect: suppress charm/fear (restored later) or pacify (SURRENDERED flag until harmed) |
| darkness | 2 | zone override: dispels lights of <=2nd level |
| darkvision | 2 | effect |
| enhance_ability | 2 | hook + effect: advantage, Bear's Endurance temp HP, Cat's Grace fall (<=20 ft) |
| enlarge_reduce | 2 | hook + `enlarge`/`reduce` (size, STR adv/dis, +-1d4) |
| find_traps | 2 | hook: reveals traps/trapped chests in LOS within 120 ft |
| flame_blade | 2 | hook + effect: `flame_blade_attack`, light, +1d6 per two slots |
| flaming_sphere | 2 | hook + zone: DEX save at end of turn when adjacent, `flaming_sphere_move` with ramming |
| gust_of_wind | 2 | hook + zone: push 15 ft, difficult terrain (all directions), `gust_of_wind_redirect` |
| heat_metal | 2 | hook: needs worn/held metal; 2d8 fire, CON save to drop else disadvantage, `heat_metal_pulse` bonus action |
| hold_person | 2 | generic; `hold_person` effect undoes paralysis of non-humanoids |
| invisibility | 2 | effect: ends on attack/cast |
| knock | 2 | hook: unlocks door/chest, alerts hostiles |
| lesser_restoration | 2 | hook: ends paralyzed>blinded>poisoned>deafened (or chosen) |
| levitate | 2 | effect: z=4, speed 20; willing allies still roll the save |
| magic_weapon | 2 | hook + effect: +1/+2/+3 by slot |
| mirror_image | 2 | effect: redirects attack rolls (d20 6+/8+/11+), images lost on hit |
| misty_step | 2 | base hook |
| moonbeam | 2 | zone: upcast dice, shapechanger disadvantage, `moonbeam_move` |
| pass_without_trace | 2 | effect: +10 Stealth for the caster's allies |
| prayer_of_healing | 2 | generic (exploration only) |
| protection_from_poison | 2 | hook: cures poison, resistance, advantage |
| ray_of_enfeeblement | 2 | effect: halves STR weapon damage, CON save at turn end |
| scorching_ray | 2 | generic |
| see_invisibility | 2 | effect sets `senses['see_invisible']` (engine must honour it) |
| shatter | 2 | generic |
| silence | 2 | zone: deafened ward immune to thunder; verbal spells blocked |
| spider_climb | 2 | effect |
| spike_growth | 2 | zone: 2d4 per square |
| spiritual_weapon | 2 | hook + zone (attached to caster) + `spiritual_weapon_attack`; the weapon flies next to the target: distance 1 and no cover (melee vs prone has advantage) |
| suggestion | 2 | hook: follow/flee/stand_down/drop_weapon, 8 h |
| warding_bond | 2 | hook + effect |
| web | 2 | zone: fire damage to a creature in it ignites it (burns away in a round, 2d4 to creatures starting their turn there); needs the engine call of `ZoneImpl.on_damage` (requested) |
| animate_dead | 3 | hook: exploration only; skeleton/zombie from a humanoid corpse, 24 h, +2 undead/slot above 3rd |
| beacon_of_hope | 3 | effect (+ heal/death-save wrappers) |
| bestow_curse | 3 | hook + effect: 4 options; 5th+ slot: no concentration |
| blink | 3 | hook (self effect): d20 at turn end, 11+ -> `ethereal` until next ROUND_START (not own turn) |
| call_lightning | 3 | hook + zone + `call_lightning_strike` |
| conjure_animals | 3 | hook: options 1xCR2/2xCR1/4xCR1/2/8xCR1/4, x2/x3/x4 at slots 5/7/9, group initiative, dismissed with concentration |
| counterspell | 3 | base reaction hook |
| daylight | 3 | hook + zone: dispels darkness <=3rd |
| dispel_magic | 3 | hook: effects, linked conditions, attached zones, summons; DC 10+level above slot level. Conditions of non-concentration spells are remembered (`fstate['spell_conds']`) and dispellable |
| fear | 3 | hook + `afraid`: drop weapons, Dash away, save at turn end only out of sight |
| fireball | 3 | generic |
| fly | 3 | generic |
| gaseous_form | 3 | effect: fly 10, resistance, cannot act |
| haste | 3 | hook (generic) + base haste/lethargy effects |
| hypnotic_pattern | 3 | hook + `hypnotized`: charmed+incapacitated+speed 0; damage or Help wakes |
| lightning_bolt | 3 | generic |
| mass_healing_word | 3 | generic |
| protection_from_energy | 3 | hook: 5 types |
| remove_curse | 3 | hook: ends bestow_curse/cursed_gold |
| revivify | 3 | base hook |
| sleet_storm | 3 | hook + zone: prone, concentration disadvantage, douses flames |
| slow | 3 | effect: speed/2, -2 AC & DEX, no reactions, one attack, WIS save at turn end; one action OR bonus action (`economy_mods`), one attack per turn, d20 11+ delays a 1-action spell to the next turn |
| spirit_guardians | 3 | generic zone |
| stinking_cloud | 3 | zone: poisoned + `retching` (action lost) |
| vampiric_touch | 3 | hook + `vampiric_touch_attack`; heals half |
| wind_wall | 3 | hook + zone: damage on creation; ordinary ranged weapon attacks across it auto-miss (engine `blocks_projectiles`), gas does not cross |
| banishment | 4 | hook + `banished` (extraplanar natives stay away after a full minute) |
| greater_invisibility | 4 | generic (+ invisibility effect) |
| guardian_of_faith | 4 | hook + zone (enemies only, 60 damage cap) |
| ice_storm | 4 | zone: cast-time damage (2d8+4d6) + difficult terrain |
| cone_of_cold | 5 | generic |

## Items
Effects in `item_effects.py` (damage_resistance, antitoxin, invisible, climbing_speed, set_ability(_min), bracers_of_defense, elvenkind, silent_steps, displacement, free_action, mind_shielding, jump_spell, war_mage, cursed_gold, leverage, stabilize_no_check, oiled, potion_of_poison, caltrops/ball_bearings zones). Hooks in `item_hooks.py`: alchemist's fire, oil flask (auto-target nearest hostile), caltrops, ball bearings, potion of poison, wearable `use_` = put on; `sync_item_effects` / `equip_magic_item` for rings, cloaks, amulets, healer's kit, crowbar, war-mage wand.
