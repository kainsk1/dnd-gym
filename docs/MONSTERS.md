# Emberdeep monster roster (SRD 5.1)

Generated from `emberdeep/data/monsters*.py`. 175 monsters. Source of truth is the code; this table is a reference.

## Conventions

* Attack damage dice include the flat modifier (`1d6+2`, `add_mod=False`); `to_hit` is the total SRD bonus.
* Hit points: `hp_dice` carries the CON bonus (`2d8+2`); `hp` is the floor of the average. A test checks both.
* `resist_nonmagical_physical=True` is "bludgeoning, piercing and slashing from nonmagical attacks". If the SRD adds an exception, `trait_params["physical_exception"]` is `"silvered"` (wight, wraith, imp, quasit) or `"adamantine"` (gargoyle).
* Immunity to nonmagical physical damage (werewolf, wererat) is the trait `immune_nonmagical_physical` plus `trait_params["physical_exception"] = "silvered"`.
* Swarms have plain B/P/S resistance (tuple), not the nonmagical flag. Exhaustion immunity is the trait `immune_exhaustion` (there is no Condition for exhaustion).
* Breath weapons and similar are `SaveActionDef` with `recharge=5` ("Recharge 5-6"; `4` for "4-6", `6` for "6"). Dragon wyrmlings with two breaths share one recharge: both actions carry the tag `shared_recharge:breath`.
* Poison riders: `on_hit_save` + `on_hit_save_damage`. Spider/wasp/centipede-style poison also carries `on_hit_effect = poison_paralysis_at_zero`.
* Multiattack is `multiattack`: a tuple of alternative sequences of action ids. Forms (lycanthropes, imp, quasit, mimic) are described in `trait_params["shapechanger"]`.
* Tags on actions (e.g. `grapple_medium_or_smaller`, `one_target_at_a_time`, `large_or_smaller`, `web_ac10_hp5`) are advisory hints for the engine.
* Parry reactions are `SpecialActionDef(hook="parry", params={"bonus_ac": N})` in `reactions`.
* Loot `equipment` ids are SRD weapon/armor ids owned by `data/weapons.py` / `data/armor.py` (e.g. `chain_shirt`, `studded_leather`, `hide`).
* Treasure: `treasure_gp` is own design by CR band (beasts/oozes/plants/constructs/elementals carry none).
* Not included (not SRD 5.1 or unsure): duergar, troglodyte, banshee, poltergeist, pixie, winged kobold, giant ape (CR 7), archmage (CR 12) and all non-SRD MM creatures.
* Spells not in `data/spell_ids.py` were dropped from caster lists: druidcraft (druid, dryad), prestidigitation (mage), speak_with_animals and animal_messenger (druid), nondetection and disguise_self (svirfneblin).

## Encounter budget (own design)

Per character: `round5(40 * level^1.5)` is the *medium* budget; easy x0.5, hard x1.5, deadly x2.2. Raw XP (SRD table) is multiplied by a group-size factor: 1 -> 1.0, 2 -> 1.4, 3-4 -> 1.8, 5-6 -> 2.2, 7-9 -> 2.6, 10-14 -> 3.0, 15+ -> 3.5. `build_encounter` fills a group to 0.75-1.1 x budget.

## Monsters

| id | CR | XP | AC | HP | Speed | Profile | Morale | Notable traits |
|---|---|---|---|---|---|---|---|---|
| `bat` | 0 | 10 | 12 | 1 | 5/fly 30 | coward | 0.5 |  |
| `cat` | 0 | 10 | 12 | 2 | 40/climb 30 | coward | 0.5 |  |
| `commoner` | 0 | 10 | 10 | 4 | 30 | coward | 0.5 |  |
| `giant_fire_beetle` | 0 | 10 | 13 | 4 | 30 | beast | 0.2 |  |
| `hawk` | 0 | 10 | 13 | 1 | 10/fly 60 | coward | 0.5 |  |
| `homunculus` | 0 | 10 | 13 | 5 | 20/fly 40 | coward | 0.4 |  |
| `hyena` | 0 | 10 | 11 | 5 | 50 | beast | 0.4 | pack_tactics |
| `jackal` | 0 | 10 | 12 | 3 | 40 | beast | 0.4 | pack_tactics |
| `lemure` | 0 | 10 | 7 | 13 | 15 | brute | 0 |  |
| `owl` | 0 | 10 | 11 | 1 | 5/fly 60 | coward | 0.5 |  |
| `rat` | 0 | 10 | 10 | 1 | 20 | coward | 0.5 |  |
| `raven` | 0 | 10 | 12 | 1 | 10/fly 50 | coward | 0.5 |  |
| `shrieker` | 0 | 10 | 5 | 13 | 0 | ambusher | 0 | false_appearance |
| `vulture` | 0 | 10 | 10 | 5 | 10/fly 50 | coward | 0.5 | pack_tactics |
| `bandit` | 1/8 | 25 | 12 | 11 | 30 | skirmisher | 0.3 |  |
| `blood_hawk` | 1/8 | 25 | 12 | 7 | 10/fly 60 | beast | 0.3 | pack_tactics |
| `cultist` | 1/8 | 25 | 12 | 9 | 30 | brute | 0.1 |  |
| `giant_crab` | 1/8 | 25 | 15 | 13 | 30/swim 30 | beast | 0.2 |  |
| `giant_rat` | 1/8 | 25 | 12 | 7 | 30 | beast | 0.4 | pack_tactics |
| `guard` | 1/8 | 25 | 16 | 11 | 30 | brute | 0.3 |  |
| `kobold` | 1/8 | 25 | 12 | 5 | 30 | skirmisher | 0.5 | sunlight_sensitivity, pack_tactics |
| `mastiff` | 1/8 | 25 | 12 | 5 | 40 | beast | 0.3 |  |
| `merfolk` | 1/8 | 25 | 11 | 11 | 10/swim 40 | skirmisher | 0.3 |  |
| `mule` | 1/8 | 25 | 10 | 11 | 30 | coward | 0.5 |  |
| `noble` | 1/8 | 25 | 15 | 9 | 30 | coward | 0.5 |  |
| `poisonous_snake` | 1/8 | 25 | 13 | 5 | 30/swim 30 | ambusher | 0.2 |  |
| `pony` | 1/8 | 25 | 10 | 11 | 40 | coward | 0.5 |  |
| `stirge` | 1/8 | 25 | 14 | 2 | 10/fly 40 | beast | 0.2 |  |
| `tribal_warrior` | 1/8 | 25 | 12 | 11 | 30 | brute | 0.3 | pack_tactics |
| `acolyte` | 1/4 | 50 | 10 | 9 | 30 | support | 0.25 | spellcaster |
| `axe_beak` | 1/4 | 50 | 11 | 19 | 50 | beast | 0.3 |  |
| `blink_dog` | 1/4 | 50 | 13 | 22 | 40 | skirmisher | 0.3 | recharge |
| `boar` | 1/4 | 50 | 11 | 11 | 40 | beast | 0.2 | charge, relentless |
| `constrictor_snake` | 1/4 | 50 | 12 | 13 | 30/swim 30 | ambusher | 0.2 |  |
| `draft_horse` | 1/4 | 50 | 10 | 19 | 40 | coward | 0.5 |  |
| `dretch` | 1/4 | 50 | 11 | 18 | 20 | brute | 0.15 |  |
| `drow` | 1/4 | 50 | 15 | 13 | 30 | skirmisher | 0.25 | sunlight_sensitivity, innate_spellcasting, spellcaster |
| `elk` | 1/4 | 50 | 10 | 13 | 50 | beast | 0.3 | charge |
| `flying_sword` | 1/4 | 50 | 17 | 17 | 0/fly 50 (hover) | skirmisher | 0 | antimagic_susceptibility, false_appearance |
| `giant_badger` | 1/4 | 50 | 10 | 13 | 30/burrow 10 | beast | 0.2 |  |
| `giant_bat` | 1/4 | 50 | 13 | 22 | 10/fly 60 | beast | 0.3 |  |
| `giant_centipede` | 1/4 | 50 | 13 | 18 | 30/climb 30 | beast | 0.2 |  |
| `giant_frog` | 1/4 | 50 | 11 | 18 | 30/swim 30 | ambusher | 0.2 |  |
| `giant_owl` | 1/4 | 50 | 12 | 19 | 5/fly 60 | beast | 0.25 |  |
| `giant_poisonous_snake` | 1/4 | 50 | 14 | 11 | 30/swim 30 | ambusher | 0.2 |  |
| `giant_wolf_spider` | 1/4 | 50 | 13 | 11 | 40/climb 40 | ambusher | 0.25 | spider_climb |
| `goblin` | 1/4 | 50 | 15 | 7 | 30 | skirmisher | 0.4 | nimble_escape |
| `grimlock` | 1/4 | 50 | 11 | 11 | 30 | ambusher | 0.3 |  |
| `panther` | 1/4 | 50 | 12 | 13 | 50/climb 40 | ambusher | 0.25 | pounce |
| `pseudodragon` | 1/4 | 50 | 13 | 7 | 15/fly 60 | skirmisher | 0.3 | magic_resistance |
| `riding_horse` | 1/4 | 50 | 10 | 13 | 60 | coward | 0.5 |  |
| `skeleton` | 1/4 | 50 | 13 | 13 | 30 | brute | 0 |  |
| `sprite` | 1/4 | 50 | 15 | 2 | 10/fly 40 | skirmisher | 0.3 |  |
| `swarm_of_bats` | 1/4 | 50 | 12 | 22 | 0/fly 30 | beast | 0 | swarm |
| `swarm_of_rats` | 1/4 | 50 | 10 | 24 | 30 | beast | 0 | swarm |
| `violet_fungus` | 1/4 | 50 | 5 | 18 | 5 | ambusher | 0 | false_appearance |
| `wolf` | 1/4 | 50 | 13 | 11 | 40 | beast | 0.3 | pack_tactics |
| `zombie` | 1/4 | 50 | 8 | 22 | 30 | brute | 0 | undead_fortitude |
| `ape` | 1/2 | 100 | 12 | 19 | 30/climb 30 | beast | 0.3 |  |
| `black_bear` | 1/2 | 100 | 11 | 19 | 40/climb 30 | beast | 0.3 |  |
| `cockatrice` | 1/2 | 100 | 11 | 27 | 20/fly 40 | beast | 0.2 |  |
| `crocodile` | 1/2 | 100 | 12 | 19 | 20/swim 30 | ambusher | 0.2 |  |
| `darkmantle` | 1/2 | 100 | 11 | 22 | 10 | ambusher | 0.2 | false_appearance |
| `giant_goat` | 1/2 | 100 | 11 | 19 | 40 | beast | 0.25 | charge |
| `giant_wasp` | 1/2 | 100 | 12 | 13 | 10/fly 50 | beast | 0.2 |  |
| `gnoll` | 1/2 | 100 | 15 | 22 | 30 | brute | 0.2 | rampage |
| `gray_ooze` | 1/2 | 100 | 8 | 22 | 10/climb 10 | ambusher | 0 | amorphous, false_appearance |
| `hobgoblin` | 1/2 | 100 | 18 | 11 | 30 | leader | 0.15 | martial_advantage |
| `lizardfolk` | 1/2 | 100 | 15 | 22 | 30/swim 30 | brute | 0.2 |  |
| `orc` | 1/2 | 100 | 13 | 15 | 30 | brute | 0.15 | aggressive |
| `reef_shark` | 1/2 | 100 | 12 | 22 | 0/swim 40 | beast | 0.2 | pack_tactics |
| `rust_monster` | 1/2 | 100 | 14 | 27 | 40 | beast | 0.2 |  |
| `sahuagin` | 1/2 | 100 | 12 | 22 | 30/swim 40 | brute | 0.2 |  |
| `satyr` | 1/2 | 100 | 14 | 31 | 30 | skirmisher | 0.3 | magic_resistance |
| `scout` | 1/2 | 100 | 13 | 16 | 30 | archer | 0.3 |  |
| `shadow` | 1/2 | 100 | 12 | 16 | 30 | ambusher | 0 | amorphous |
| `svirfneblin` | 1/2 | 100 | 15 | 16 | 30 | skirmisher | 0.25 | innate_spellcasting, spellcaster |
| `swarm_of_insects` | 1/2 | 100 | 12 | 22 | 20/climb 20 | beast | 0 | swarm |
| `thug` | 1/2 | 100 | 11 | 32 | 30 | brute | 0.25 | pack_tactics |
| `warhorse` | 1/2 | 100 | 11 | 19 | 60 | beast | 0.3 |  |
| `warhorse_skeleton` | 1/2 | 100 | 13 | 22 | 60 | brute | 0 |  |
| `worg` | 1/2 | 100 | 13 | 26 | 50 | beast | 0.3 |  |
| `animated_armor` | 1 | 200 | 18 | 33 | 30 | brute | 0 | antimagic_susceptibility, false_appearance |
| `brass_dragon_wyrmling` | 1 | 200 | 16 | 16 | 30/burrow 15/fly 60 | boss | 0.1 | recharge |
| `brown_bear` | 1 | 200 | 11 | 34 | 40/climb 30 | beast | 0.25 |  |
| `bugbear` | 1 | 200 | 16 | 27 | 30 | ambusher | 0.25 | brute |
| `copper_dragon_wyrmling` | 1 | 200 | 16 | 22 | 30/climb 30/fly 60 | boss | 0.1 | recharge |
| `death_dog` | 1 | 200 | 12 | 39 | 40 | beast | 0.2 |  |
| `dire_wolf` | 1 | 200 | 14 | 37 | 50 | beast | 0.3 | pack_tactics |
| `dryad` | 1 | 200 | 11 | 22 | 30 | support | 0.3 | magic_resistance, innate_spellcasting, spellcaster |
| `ghoul` | 1 | 200 | 12 | 22 | 30 | brute | 0 |  |
| `giant_eagle` | 1 | 200 | 13 | 26 | 10/fly 80 | beast | 0.2 |  |
| `giant_hyena` | 1 | 200 | 12 | 45 | 50 | beast | 0.25 | rampage |
| `giant_spider` | 1 | 200 | 14 | 26 | 30/climb 30 | ambusher | 0.2 | spider_climb, recharge |
| `giant_toad` | 1 | 200 | 11 | 39 | 20/swim 40 | ambusher | 0.2 |  |
| `giant_vulture` | 1 | 200 | 10 | 22 | 10/fly 60 | beast | 0.3 | pack_tactics |
| `harpy` | 1 | 200 | 11 | 38 | 20/fly 40 | ambusher | 0.3 |  |
| `hippogriff` | 1 | 200 | 11 | 19 | 40/fly 60 | beast | 0.25 |  |
| `imp` | 1 | 200 | 13 | 10 | 20/fly 40 | skirmisher | 0.3 | shapechanger, magic_resistance |
| `lion` | 1 | 200 | 12 | 26 | 50 | beast | 0.2 | pack_tactics, pounce |
| `quasit` | 1 | 200 | 13 | 7 | 40 | skirmisher | 0.3 | shapechanger, magic_resistance |
| `specter` | 1 | 200 | 12 | 22 | 0/fly 50 (hover) | ambusher | 0 | incorporeal_movement, sunlight_sensitivity |
| `spy` | 1 | 200 | 12 | 27 | 30 | ambusher | 0.4 | sneak_attack |
| `tiger` | 1 | 200 | 12 | 37 | 40 | ambusher | 0.2 | pounce |
| `ankheg` | 2 | 450 | 14 | 39 | 30/burrow 10 | ambusher | 0.2 | recharge |
| `bandit_captain` | 2 | 450 | 15 | 65 | 30 | leader | 0.2 |  |
| `berserker` | 2 | 450 | 13 | 67 | 30 | brute | 0 | reckless |
| `black_dragon_wyrmling` | 2 | 450 | 17 | 33 | 30/fly 60/swim 30 | boss | 0.1 | recharge |
| `bronze_dragon_wyrmling` | 2 | 450 | 17 | 32 | 30/fly 60/swim 30 | boss | 0.1 | recharge |
| `centaur` | 2 | 450 | 12 | 45 | 50 | archer | 0.25 | charge |
| `cult_fanatic` | 2 | 450 | 13 | 33 | 30 | caster | 0.1 | spellcaster |
| `druid` | 2 | 450 | 11 | 27 | 30 | caster | 0.25 | spellcaster |
| `ettercap` | 2 | 450 | 13 | 44 | 30/climb 30 | ambusher | 0.2 | spider_climb, recharge |
| `gargoyle` | 2 | 450 | 15 | 52 | 30/fly 60 | ambusher | 0.1 | false_appearance |
| `gelatinous_cube` | 2 | 450 | 6 | 84 | 15 | brute | 0 |  |
| `ghast` | 2 | 450 | 13 | 36 | 30 | brute | 0 |  |
| `giant_boar` | 2 | 450 | 12 | 42 | 40 | beast | 0.15 | charge, relentless |
| `giant_constrictor_snake` | 2 | 450 | 12 | 60 | 30/swim 30 | ambusher | 0.2 |  |
| `giant_elk` | 2 | 450 | 14 | 42 | 60 | beast | 0.25 | charge |
| `gibbering_mouther` | 2 | 450 | 9 | 67 | 10/swim 10 | brute | 0 | recharge |
| `green_dragon_wyrmling` | 2 | 450 | 17 | 38 | 30/fly 60/swim 30 | boss | 0.1 | recharge |
| `grick` | 2 | 450 | 14 | 27 | 30/climb 30 | ambusher | 0.2 |  |
| `griffon` | 2 | 450 | 12 | 59 | 30/fly 80 | beast | 0.2 |  |
| `mimic` | 2 | 450 | 12 | 58 | 15 | ambusher | 0 | shapechanger, false_appearance |
| `minotaur_skeleton` | 2 | 450 | 12 | 67 | 30 | brute | 0 | charge |
| `ochre_jelly` | 2 | 450 | 8 | 45 | 10/climb 10 | brute | 0 | amorphous, spider_climb, split |
| `ogre` | 2 | 450 | 11 | 59 | 30 | brute | 0.2 |  |
| `ogre_zombie` | 2 | 450 | 8 | 85 | 30 | brute | 0 | undead_fortitude |
| `polar_bear` | 2 | 450 | 12 | 42 | 40/swim 30 | beast | 0.2 |  |
| `priest` | 2 | 450 | 13 | 27 | 30 | support | 0.25 | spellcaster |
| `rhinoceros` | 2 | 450 | 11 | 45 | 40 | beast | 0.2 | charge |
| `rug_of_smothering` | 2 | 450 | 12 | 33 | 10 | ambusher | 0 | antimagic_susceptibility, false_appearance |
| `saber_toothed_tiger` | 2 | 450 | 12 | 52 | 40 | ambusher | 0.2 | pounce |
| `silver_dragon_wyrmling` | 2 | 450 | 17 | 45 | 30/fly 60 | boss | 0.1 | recharge |
| `wererat` | 2 | 450 | 12 | 33 | 30 | ambusher | 0.25 | shapechanger, immune_nonmagical_physical |
| `white_dragon_wyrmling` | 2 | 450 | 16 | 32 | 30/burrow 15/fly 60/swim 30 | boss | 0.1 | recharge |
| `will_o_wisp` | 2 | 450 | 19 | 22 | 0/fly 50 (hover) | ambusher | 0 | incorporeal_movement |
| `basilisk` | 3 | 700 | 15 | 52 | 20 | ambusher | 0.2 | petrifying_gaze |
| `blue_dragon_wyrmling` | 3 | 700 | 17 | 52 | 30/burrow 15/fly 60/swim 30 | boss | 0.1 | recharge |
| `doppelganger` | 3 | 700 | 14 | 52 | 30 | ambusher | 0.3 | shapechanger |
| `giant_scorpion` | 3 | 700 | 15 | 52 | 30 | ambusher | 0.2 |  |
| `gold_dragon_wyrmling` | 3 | 700 | 17 | 60 | 30/fly 60/swim 30 | boss | 0.1 | recharge |
| `green_hag` | 3 | 700 | 17 | 82 | 30 | caster | 0.2 | innate_spellcasting, spellcaster |
| `hell_hound` | 3 | 700 | 15 | 45 | 50 | beast | 0.1 | pack_tactics, recharge |
| `knight` | 3 | 700 | 18 | 52 | 30 | leader | 0.1 |  |
| `manticore` | 3 | 700 | 14 | 68 | 30/fly 50 | archer | 0.2 |  |
| `minotaur` | 3 | 700 | 14 | 76 | 30 | brute | 0 | charge, reckless |
| `mummy` | 3 | 700 | 11 | 58 | 30 | brute | 0 |  |
| `owlbear` | 3 | 700 | 13 | 59 | 30 | beast | 0.1 |  |
| `phase_spider` | 3 | 700 | 13 | 32 | 30/climb 30 | ambusher | 0.2 | spider_climb |
| `veteran` | 3 | 700 | 17 | 58 | 30 | brute | 0.15 |  |
| `werewolf` | 3 | 700 | 11 | 58 | 30 | brute | 0.2 | shapechanger, immune_nonmagical_physical |
| `wight` | 3 | 700 | 14 | 45 | 30 | leader | 0 | sunlight_sensitivity |
| `winter_wolf` | 3 | 700 | 13 | 75 | 50 | beast | 0.2 | pack_tactics, recharge |
| `black_pudding` | 4 | 1100 | 7 | 85 | 20/climb 20 | brute | 0 | amorphous, spider_climb, split |
| `ettin` | 4 | 1100 | 12 | 85 | 30 | brute | 0.15 |  |
| `ghost` | 4 | 1100 | 11 | 45 | 0/fly 40 (hover) | boss | 0 | incorporeal_movement, recharge |
| `red_dragon_wyrmling` | 4 | 1100 | 17 | 75 | 30/climb 30/fly 60 | boss | 0.1 | recharge |
| `air_elemental` | 5 | 1800 | 15 | 90 | 0/fly 90 (hover) | brute | 0 | recharge |
| `bulette` | 5 | 1800 | 17 | 94 | 40/burrow 40 | beast | 0.2 |  |
| `earth_elemental` | 5 | 1800 | 17 | 126 | 30/burrow 30 | brute | 0 |  |
| `fire_elemental` | 5 | 1800 | 13 | 102 | 50 | brute | 0 |  |
| `gladiator` | 5 | 1800 | 16 | 112 | 30 | brute | 0.1 | brute |
| `hill_giant` | 5 | 1800 | 13 | 105 | 30 | brute | 0.15 |  |
| `otyugh` | 5 | 1800 | 14 | 114 | 30 | ambusher | 0.1 |  |
| `troll` | 5 | 1800 | 15 | 84 | 30 | brute | 0.1 | regeneration |
| `vampire_spawn` | 5 | 1800 | 15 | 82 | 30 | boss | 0.1 | regeneration, spider_climb |
| `water_elemental` | 5 | 1800 | 14 | 114 | 30/swim 90 | brute | 0 | recharge |
| `wraith` | 5 | 1800 | 13 | 67 | 0/fly 60 (hover) | ambusher | 0 | incorporeal_movement, sunlight_sensitivity |
| `chimera` | 6 | 2300 | 14 | 114 | 30/fly 60 | boss | 0.1 | recharge |
| `mage` | 6 | 2300 | 12 | 40 | 30 | caster | 0.25 | spellcaster |
| `wyvern` | 6 | 2300 | 13 | 110 | 20/fly 80 | boss | 0.1 |  |
| `assassin` | 8 | 3900 | 15 | 78 | 30 | ambusher | 0.3 | sneak_attack |
| `young_red_dragon` | 10 | 5900 | 18 | 178 | 40/climb 40/fly 80 | boss | 0 | recharge |
| `adult_white_dragon` | 13 | 10000 | 18 | 200 | 40/burrow 30/fly 80/swim 40 | boss | 0 | legendary_resistance, legendary x3, recharge |

## Hook ids phase-2 engineers must implement

### traits

`aberrant_ground`, `adhesive`, `aggressive`, `air_form`, `ambusher`, `amorphous`, `amphibious`, `antimagic_susceptibility`, `assassinate`, `beast_of_burden`, `blind_senses`, `blood_frenzy`, `brave`, `brute`, `charge`, `corrode_metal`, `corrosive_form`, `cunning_action`, `damage_transfer`, `dark_devotion`, `devils_sight`, `divine_eminence`, `earth_glide`, `echolocation`, `ephemeral`, `ethereal_sight`, `evasion`, `false_appearance`, `fey_ancestry`, `fire_form`, `flyby`, `freeze`, `gibbering`, `gnome_cunning`, `grappler`, `hellish_rejuvenation`, `hold_breath`, `ice_walk`, `illumination`, `immune_exhaustion`, `immune_nonmagical_physical`, `incorporeal_movement`, `innate_spellcasting`, `iron_scent`, `keen_hearing`, `keen_hearing_and_sight`, `keen_hearing_and_smell`, `keen_senses`, `keen_sight`, `keen_sight_and_smell`, `keen_smell`, `labyrinthine_recall`, `legendary_resistance`, `limited_amphibiousness`, `limited_telepathy`, `magic_resistance`, `martial_advantage`, `mimicry`, `nimble_escape`, `ooze_cube`, `pack_tactics`, `petrifying_gaze`, `pounce`, `rampage`, `reckless`, `regeneration`, `relentless`, `running_leap`, `rust_metal`, `shadow_stealth`, `shapechanger`, `shark_telepathy`, `siege_monster`, `sneak_attack`, `snow_camouflage`, `speak_with_beasts_and_plants`, `spider_climb`, `split`, `standing_leap`, `stench`, `stone_camouflage`, `sunlight_sensitivity`, `sunlight_weakness`, `sure_footed`, `surprise_attack`, `swarm`, `tail_spike_regrowth`, `telepathic_bond`, `trampling_charge`, `transparent`, `tree_stride`, `turning_defiance`, `two_heads`, `undead_fortitude`, `vampire_weaknesses`, `variable_illumination`, `wakeful`, `water_breathing`, `water_form`, `water_susceptibility`, `web_sense`, `web_walker`

### specials

`aggressive`, `consume_life`, `corrode_metal`, `create_specter`, `cunning_action`, `darkness_aura`, `divine_eminence`, `engulf`, `ethereal_jaunt`, `etherealness`, `grapple_attack`, `heart_sight`, `illusory_appearance`, `invisibility`, `invisible_passage`, `leadership`, `nimble_escape`, `parry`, `perception_check`, `possession`, `rampage`, `read_thoughts`, `shriek`, `swallow`, `teleport`

### effects

`aging`, `attach_smother`, `blood_drain_attach`, `burning`, `corrode_armor`, `dreadful_glare`, `engulfed`, `flung`, `luring_song`, `lycanthropy`, `max_hp_reduction`, `mummy_rot`, `petrification`, `poison_paralysis_at_zero`, `push`, `sleep_poison`, `slowing_breath`, `smother`, `strength_drain`, `wasting_disease`, `weakened`

## Encounter groups

| id | entries | leaders |
|---|---|---|
| `goblin_warband` | goblin 3-10, worg 1-4, hobgoblin 1-4, bugbear 1-3 | hobgoblin, bugbear, ogre |
| `kobold_den` | kobold 3-12, giant_rat 1-6, swarm_of_rats 1-2, giant_centipede 1-2 | red_dragon_wyrmling, black_dragon_wyrmling, giant_spider |
| `orc_raiders` | orc 2-10, worg 1-4, berserker 1-2, gnoll 1-3 | ogre, ettin, berserker |
| `gnoll_pack` | gnoll 2-8, hyena 1-4, giant_hyena 1-2 | ogre, giant_hyena |
| `bandits` | bandit 2-10, thug 1-4, scout 1-3, mastiff 1-2, spy 1-1 | bandit_captain, veteran, berserker |
| `cult` | cultist 3-10, acolyte 1-3, thug 1-2, imp 1-1, quasit 1-1 | cult_fanatic, priest, mage |
| `crypt` | skeleton 2-10, zombie 1-8, warhorse_skeleton 1-2, ghoul 1-4, shadow 1-3, specter 1-2, ogre_zombie 1-2, minotaur_skeleton 1-1 | wight, ghast, mummy, wraith, vampire_spawn, ghost |
| `ghoul_pack` | ghoul 2-8, ghast 1-2, shadow 1-2 | ghast, wight |
| `spider_nest` | giant_wolf_spider 2-8, giant_spider 1-4, swarm_of_insects 1-1, giant_centipede 1-3 | ettercap, phase_spider, giant_spider |
| `beasts_forest` | wolf 1-6, black_bear 1-2, boar 1-3, giant_badger 1-3, panther 1-2, giant_owl 1-1, brown_bear 1-2, giant_elk 1-1, blood_hawk 1-4, giant_frog 1-3 | owlbear, dire_wolf, tiger, giant_boar |
| `wolves` | wolf 2-10, dire_wolf 1-4, worg 1-3 | dire_wolf, winter_wolf |
| `oozes` | gray_ooze 1-4, ochre_jelly 1-2, gelatinous_cube 1-1, swarm_of_rats 1-2, violet_fungus 1-2 | black_pudding, gelatinous_cube |
| `elementals` | fire_elemental 1-3, water_elemental 1-3, air_elemental 1-3, earth_elemental 1-3, gargoyle 1-3 |  |
| `giants` | ogre 1-4, hill_giant 1-2, ettin 1-2, troll 1-2, worg 1-2 | hill_giant, troll, ettin |
| `dragon_lair_red` | kobold 3-10, hell_hound 1-3, fire_elemental 1-1, ogre 1-2, gnoll 1-3 | red_dragon_wyrmling, young_red_dragon |
| `dragon_lair_white` | kobold 3-8, winter_wolf 1-3, polar_bear 1-2, ogre 1-2, wolf 1-4 | white_dragon_wyrmling, adult_white_dragon |
| `lizardfolk_tribe` | lizardfolk 3-8, crocodile 1-3, giant_frog 1-3, constrictor_snake 1-2 | green_hag, giant_constrictor_snake, black_dragon_wyrmling |
| `sahuagin_raid` | sahuagin 3-10, reef_shark 1-4, giant_crab 1-2 | water_elemental, sahuagin |
| `drow_patrol` | drow 3-8, giant_spider 1-3, giant_wolf_spider 1-2, svirfneblin 1-2 | mage, priest, cult_fanatic |
| `underdark` | grimlock 2-8, drow 1-4, svirfneblin 1-3, gibbering_mouther 1-1, rust_monster 1-2, grick 1-3, darkmantle 1-3, ettercap 1-1, giant_spider 1-2 | otyugh, gelatinous_cube, phase_spider, ettin |
| `demons` | dretch 3-10, quasit 1-3, gargoyle 1-2 | quasit, gargoyle |
| `devils` | lemure 3-10, imp 1-3, hell_hound 1-3 | imp, hell_hound |
