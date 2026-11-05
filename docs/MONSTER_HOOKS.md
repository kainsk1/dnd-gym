# Monster hooks (phase 2, monster-hooks)

Implementation of every hook id referenced by `emberdeep/data/monsters*.py` (`MONSTER_HOOKS_REFERENCED`): traits, special
actions and effects. Code: `emberdeep/rules/content/monster_util.py` (helpers, `SPECIAL_VFX`), `monster_traits.py` (senses,
modifiers, resistances, regeneration...), `monster_traits_forms.py` (disguise, shapechanging, swarms, splitting, corrosion,
auras, reactions), `monster_traits_move.py` (charge / pounce / rampage), `monster_specials.py`, `monster_effects.py`.
Tests: `tests/test_monster_hooks.py` (unit, seeded / scripted dice) and `tests/test_monsters_play.py` (every monster in 20 random fights +
an ability sweep + coverage summary). Engine gaps are listed in `docs/dev/contract_requests/monster-hooks.md` ("request N" below).

## Conventions

* Trait parameters: the factory forwards `trait_params[trait_id]` (dicts as-is, scalars as `params['value']`); other top-level keys
  (`physical_exception`, `regeneration_stopped_by`, `vampire_weaknesses`, `split`, `shapechanger`...) are read through
  `monster_util.tparam(creature, key)`.
* Scratch state lives in the hook's `state` dict or `Creature.fstate['mh']` (both are copied by `World.clone`; only immutable values
  or re-assigned containers are stored).
* Lazily initialised traits (disguise, lights, auras, swarm no-heal) set themselves up in the first `on_round_start`, `on_turn_start` or
  `initiative_bonus` call.
* Effects read the carrying attack back from the event log of the current step (`monster_effects.recent_damage / recent_save`).
* Reaction-only specials (`parry`, `shriek`) are `reaction_parry` / `reaction_shriek` traits; `monster_traits_forms.attach_reaction_hooks(c)`
  attaches them (request 1).
* Every special has a renderer key in `monster_util.SPECIAL_VFX` (tested against `render.api.VFX_KEYS`).
* Registered extras (not referenced by data): traits `reaction_parry`, `reaction_shriek`; effects `mh_no_healing`, `mh_speed_zero`,
  `mh_divine_eminence`, `mh_leadership`, `mh_invisible`, `mh_ethereal`, `mh_read_thoughts`, `mh_possessed`, `mh_possessing`,
  `mh_swallowed`; zones `mh_aberrant_ground`, `mh_gibbering`, `mh_stench`, `mh_gaze`, `mh_darkness`. No effect id owned by the spell
  or item engineer is registered here (tested).


## Traits

| id | monsters | implementation | deviations |
|---|---|---|---|
| `aberrant_ground` | gibbering_mouther | Gibbering mouther: 10-ft radius of doughy difficult terrain, STR save DC 10 or speed 0. | Attached emanation zone mh_aberrant_ground. |
| `adhesive` | mimic | Object form only: a creature that touches or hits the mimic in melee is grappled by it (escape DC 13). | Triggers on melee attacks against the object-form mimic. |
| `aggressive` | orc | Dash toward a hostile as a bonus action: see the 'aggressive' special action. | Rules live in the aggressive special. |
| `air_form` | air_elemental | May enter hostile spaces / 1-inch gaps (needs engine support: see contract request). | Shares squares with hostile creatures; squeezes without penalty. |
| `ambusher` | doppelganger | Advantage on attack rolls against any creature it has surprised. |  |
| `amorphous` | black_pudding, gray_ooze, ochre_jelly, shadow | Can move through a space as narrow as 1 inch (squeezing is not modelled). | Large amorphous creatures squeeze through 1-wide corridors without penalty. |
| `amphibious` | giant_crab, giant_frog, giant_toad, green_hag +1 | Breathes air and water (underwater combat is not modelled). | No underwater / suffocation rules in the engine. |
| `antimagic_susceptibility` | animated_armor, flying_sword, rug_of_smothering | Incapacitated while in an antimagic field; dispel magic forces a CON save or 1 minute of unconsciousness. ``antimagic_dispel_check`` is the entry point for the dispel-magic spell hook (nothing else exists in the SRD spell list that creates an antimagic field at these levels). | Only the dispel-magic helper antimagic_dispel_check exists (no antimagic field at these levels). |
| `assassinate` | assassin | Advantage against creatures that haven't taken a turn yet; hits on surprised creatures are critical hits. | 'Has not taken a turn' = surprised, or later in the initiative order during the round of the assassin's first turn of the combat. |
| `beast_of_burden` | mule | Carrying capacity counts as one size larger (encumbrance is not modelled). | No encumbrance model. |
| `blind_senses` | grimlock | Grimlock: no blindsight while deafened. | Same as echolocation (no 'smell' model). |
| `blood_frenzy` | sahuagin | Advantage on melee attack rolls against any creature that doesn't have all its hit points. |  |
| `brave` | gladiator, knight | engine (`traits`):  | |
| `brute` | bugbear, gladiator | Extra weapon die is already part of the attack damage in the data. | Included in the attack data. |
| `charge` | boar, centaur, elk, giant_boar +5 | Moved at least ``min_move_ft`` straight toward a target and then hit it on the same turn: extra ``damage`` and, if ``save`` is given, a saving throw (DC ``dc``) or the ``effect`` (prone, or pushed 10 ft and prone). | 'Straight' = last run of identical steps this turn pointing at the target; attack = params['attack'] or the first melee action. |
| `corrode_metal` | gray_ooze | Any nonmagical metal weapon that hits it corrodes: after dealing damage it takes a permanent, cumulative -1 penalty to damage rolls against this creature (-5 destroys it).  The penalty is stored on the wielder (``mh_corrode``) and subtracted from damage dealt to corroding creatures only (a global weapon penalty needs ItemInstance support). | Penalty stored on `ItemInstance.penalty` (applies to all damage of that weapon; -5 destroys it). |
| `corrosive_form` | black_pudding | Black pudding: also deals 1d8 acid to a creature that hits it with a melee attack within 5 ft (or touches it). | corrode_metal + 1d8 acid to melee attackers. |
| `cunning_action` | spy | Dash/Disengage/Hide as a bonus action: see the 'cunning_action' special action. | Rules live in the special. |
| `damage_transfer` | rug_of_smothering | While grappling a creature, the rug takes half of the damage dealt to it and the grappled creature takes the rest. | Half of the damage dealt goes to the grappled creature after the rug took half. |
| `dark_devotion` | cult_fanatic, cultist | Advantage on saving throws against being charmed or frightened. | Triggered by save tags 'charm'/'fear' (spells set them; monster actions do not). |
| `devils_sight` | imp, lemure | Sees through magical darkness (needs a sense that ignores zone_dark: contract request). | Engine `devils_sight` sense pierces magical darkness. |
| `divine_eminence` | priest | Bonus action smite: see the 'divine_eminence' special action. | Rules live in the special. |
| `earth_glide` | earth_elemental | Burrows through non-magical earth/stone without disturbing it (burrowing is not modelled). | Burrowing is not modelled (map has no burrow layer). |
| `echolocation` | bat, darkmantle, giant_bat, swarm_of_bats | Cannot use its blindsight while deafened. | Blindsight is zeroed while deafened and restored afterwards. |
| `ephemeral` | will_o_wisp | Cannot wear or carry anything (informational). | Informational. |
| `ethereal_sight` | ghost | Sees 60 ft into the Ethereal Plane (informational). | Informational. |
| `evasion` | assassin | DEX saves against damage effects: no damage on a success, half on a failure. | Tracks the last DEX save of the creature in the same step to halve/negate area damage. |
| `false_appearance` | animated_armor, darkmantle, flying_sword, gargoyle +5 | While motionless it is indistinguishable from an ordinary object / plant / pool.  Modelled with the engine's Hidden status: observers need to *find* it (Search vs the disguise DC, 15 by default); moving, attacking or being damaged ends the disguise.  Also hosts the gargoyle's adamantine rule and the shrieker's shriek reaction. | Uses the engine's Hidden status with a DC (15). Lazily initialised on the first round/turn/initiative hook. |
| `fey_ancestry` | drow | engine (`traits`): Advantage on saves against being charmed; immune to magical sleep. | |
| `fire_form` | fire_elemental | A creature that hits it with a melee attack while within 5 ft takes ``touch_damage`` fire damage. | Touch damage; may enter hostile spaces: first entry per turn 1d10 fire + `burning`. |
| `flyby` | giant_owl, owl | Doesn't provoke opportunity attacks when it flies out of an enemy's reach.  The engine has no veto hook for opportunity attacks yet, so the attack is neutralised from the target's side (see the contract request). | Engine hook `avoids_opportunity` (tested). |
| `freeze` | water_elemental | Cold damage partially freezes the creature: speed reduced by 20 ft until the end of its next turn. | Speed -20 ft until the end of its next turn (uses the spell engineer's speed_penalty effect). |
| `gibbering` | gibbering_mouther | Gibbering mouther: 20-ft aura, WIS save DC 10 or randomised behaviour. | Attached emanation zone mh_gibbering. |
| `gnome_cunning` | svirfneblin | engine (`traits`): Advantage on INT, WIS and CHA saves against magic. | |
| `grappler` | mimic | Advantage on attack rolls against any creature grappled by it. |  |
| `hellish_rejuvenation` | lemure | Returns to life in the Nine Hells (no planes modelled). | No planes. |
| `hold_breath` | crocodile, lizardfolk | Can hold its breath for ``value`` minutes (suffocation is not modelled). | idem |
| `ice_walk` | adult_white_dragon | Climbs/ignores icy difficult terrain (zone/terrain-per-creature costs need engine support). | Per-creature `ignores_difficult`. |
| `illumination` | fire_elemental, giant_fire_beetle | Sheds bright light in ``bright_ft`` and dim light for ``dim_ft`` more (trait_params['illumination']). | Light radii from trait params ((bright, dim) in squares); lazily initialised. |
| `immune_exhaustion` | air_elemental, animated_armor, black_pudding, earth_elemental +20 | Immune to exhaustion (the engine has no exhaustion condition to veto, so levels are cleared). | Engine hook `veto_exhaustion`. |
| `immune_nonmagical_physical` | wererat, werewolf | Immune to bludgeoning, piercing and slashing from nonmagical attacks (``physical_exception``: silvered). | Silvered / adamantine exceptions via `DamageCtx.silvered/adamantine`. |
| `incorporeal_movement` | ghost, specter, will_o_wisp, wraith | Can move through other creatures and objects as difficult terrain; takes 1d10 force damage if it ends its turn inside an object.  (The engine cannot yet place a creature inside another's square: see the contract request; the damage rule is implemented for when it can.) | Moves through creatures (shared squares) and through walls/pillars/statues/closed doors as difficult terrain; 1d10 force if it ends its turn inside. |
| `innate_spellcasting` | drow, dryad, green_hag, svirfneblin | Spells come from MonsterDef.spellcasting (handled by the engine). | Handled by MonsterDef.spellcasting in the factory. |
| `iron_scent` | rust_monster | Tracks ferrous metal by scent (informational). | Informational. |
| `keen_hearing` | bat, giant_bat, swarm_of_bats | Keen senses: ('hearing',). | Disabled while deafened. |
| `keen_hearing_and_sight` | giant_owl, owl, scout | Keen senses: ('hearing', 'sight'). |  |
| `keen_hearing_and_smell` | blink_dog, dire_wolf, grimlock, hell_hound +6 | engine (`traits`):  | |
| `keen_senses` | pseudodragon | engine (`traits`): Proficiency-like advantage on Perception checks (monsters: keen smell/sight/hearing). | |
| `keen_sight` | blood_hawk, giant_eagle, griffon, hawk +1 | Keen senses: ('sight',). | Disabled while blinded. |
| `keen_sight_and_smell` | giant_vulture, owlbear, vulture | Keen senses: ('sight', 'smell'). |  |
| `keen_smell` | black_bear, brown_bear, cat, giant_badger +10 | Keen senses: ('smell',). |  |
| `labyrinthine_recall` | minotaur | Perfectly recalls any path it has travelled (no memory model). | No memory model. |
| `legendary_resistance` | adult_white_dragon | ``value`` times per day, a failed saving throw becomes a success. | `after_d20` / `save_mods`; no RNG peeking, a passing save spends nothing. |
| `limited_amphibiousness` | sahuagin | Breathes air and water, needs to immerse itself every 4 hours (not modelled). | idem |
| `limited_telepathy` | otyugh, pseudodragon | Telepathy within 100 ft (no telepathy channel). | No telepathy channel. |
| `magic_resistance` | dryad, imp, pseudodragon, quasit +1 | engine (`traits`): Advantage on saving throws against spells and other magical effects. | |
| `martial_advantage` | hobgoblin | Once per turn, extra ``value`` damage to a creature hit by a weapon attack if an ally is within 5 ft of it. |  |
| `mimicry` | green_hag, raven | Mimics sounds it has heard (no sound model). | No sound model. |
| `nimble_escape` | goblin | engine (`traits`): Disengage or Hide as a bonus action. | |
| `ooze_cube` | gelatinous_cube | Engulf rules live in the 'engulf' special action. | Registered with `may_share_space` while Engulf runs. |
| `pack_tactics` | blood_hawk, dire_wolf, giant_rat, giant_vulture +11 | engine (`traits`): Advantage on an attack roll if an ally of the attacker is within 5 ft of the target (and not incapacitated). | |
| `petrifying_gaze` | basilisk | Basilisk gaze (30 ft). | Attached zone mh_gaze; at the start of its turn a creature in range may avert its eyes (SPELL_CHOICE for agents; scripted when the save likely fails): immune to the gaze, disadvantage on attacks against the basilisk, until its next turn. |
| `pounce` | lion, panther, saber_toothed_tiger, tiger | Moved at least ``min_move_ft`` straight toward a creature and then hit it with ``attack``: STR save or prone; against a prone target it may make ``bonus_attack`` as a bonus action (granted as an ability for the rest of the turn). | Bonus bite is a granted bonus-action ability usable only on the pounced, prone target. |
| `rampage` | giant_hyena, gnoll | When it reduces a creature to 0 HP with a melee attack on its turn, the 'rampage' bonus action becomes available (move up to half its speed and bite). | Sets a flag at the end of the hitting action; the special action consumes it. |
| `reckless` | berserker, minotaur | Advantage on melee weapon attack rolls; attack rolls against it have advantage (always on for monsters). | Always on for monsters. |
| `regeneration` | troll, vampire_spawn | Regain ``value`` HP at the start of each turn.  Damage of the types in ``regeneration_stopped_by`` switches it off until the start of the creature's next turn; ``regeneration_stopped_in`` lists places (sunlight, running water) where it does not work.  With ``regeneration_death_if_zero_and_stopped`` (troll) the creature stays on 0 HP, unconscious, and only dies at the start of its turn if it cannot regenerate. | Replaces the engine exemplar: honours regeneration_stopped_by/_in and the troll's 'dies only if it starts a turn at 0 HP without regenerating' (stays down unconscious). |
| `relentless` | boar, giant_boar | Once per short/long rest, damage of ``threshold`` or less that would drop it to 0 HP leaves it at 1 HP. | Uses total damage taken; once per short/long rest. |
| `running_leap` | lion | Jump distances are not modelled (movement uses 5-ft squares). | Only the generic long-jump rule over pits applies (STR score, Jump spell). |
| `rust_metal` | rust_monster | Rust monster: same rule as corrode_metal. | Same as corrode_metal. |
| `shadow_stealth` | shadow | Hide as a bonus action while in dim light or darkness. | Usable in dim light or darkness (light grid). |
| `shapechanger` | doppelganger, imp, mimic, quasit +2 | Polymorph as an action into the forms of ``trait_params['shapechanger']['forms']``.  The true form is 'humanoid' for lycanthropes, the monster's own id for imps / quasits, otherwise the amorphous form (mimic, doppelganger).  Per-form ``ac`` / ``speed`` overrides apply.  Attacks restricted to some forms by their tags switch the form for free when used (deviation: the rules charge an action) so no action is ever wasted. | Form-restricted attacks are blocked in other forms (`ability_blocked`); changing form costs an action. Scripted lycanthropes shift to hybrid with their first combat action. |
| `shark_telepathy` | sahuagin | Telepathy with sharks within 120 ft (no telepathy channel). | No telepathy channel. |
| `siege_monster` | earth_elemental | Double damage to objects and structures (doors use hit points via force_open only). | Doors have no damage-vs-object path other than force_open. |
| `sneak_attack` | assassin, spy | Once per turn, extra ``value`` damage on a weapon hit with advantage, or when an ally of the attacker is within 5 ft of the target (not incapacitated) and the attack has no disadvantage. | Any weapon attack counts as finesse/ranged. |
| `snow_camouflage` | winter_wolf | Advantage on Stealth checks made to hide in snowy terrain. | 'Snowy terrain' = snow / ice materials. |
| `speak_with_beasts_and_plants` | dryad | Communicates with beasts and plants (no language model). | No language model. |
| `spider_climb` | black_pudding, ettercap, giant_spider, giant_wolf_spider +3 | Climbs difficult surfaces, including ceilings (climb speed in data; climbing is not modelled). | Climbing is not modelled; climb speeds are data only. |
| `split` | black_pudding, ochre_jelly | Medium-or-larger ooze subjected to lightning or slashing damage with at least ``min_hp`` HP splits into two oozes one size smaller, each with half the original's current HP (rounded down). | Triggers before damage (the jellies are immune to lightning/slashing); new ooze joins the turn order after the original. |
| `standing_leap` | bulette, giant_frog, giant_toad | Jump distances are not modelled (movement uses 5-ft squares). | Only the generic long-jump rule over pits applies (STR score, Jump spell). |
| `stench` | ghast | Ghast: 5-ft stench aura (CON DC 10, poisoned). | Attached emanation zone mh_stench (24 h immunity kept for the zone's lifetime). |
| `stone_camouflage` | grick, grimlock, svirfneblin | Advantage on Stealth checks made to hide in rocky terrain. | 'Rocky terrain' = stone/cave/cobble/rubble/bone/masonry materials. |
| `sunlight_sensitivity` | drow, kobold, specter, wight +1 | engine (`traits`): Disadvantage on attack rolls and Perception while in sunlight (bright ambient light). | |
| `sunlight_weakness` | shadow | Disadvantage on attack rolls, ability checks and saving throws while in sunlight. |  |
| `sure_footed` | giant_goat, mule | Advantage on Strength and Dexterity saves against effects that would knock it prone. | Only for saves tagged 'prone'. |
| `surprise_attack` | bugbear, doppelganger | Extra ``value`` damage when it hits a surprised creature (first round of combat). |  |
| `swarm` | swarm_of_bats, swarm_of_insects, swarm_of_rats | Swarm of Tiny creatures: half damage with its attacks when at half HP or fewer; can't regain HP or gain temp HP. (Occupying other creatures' squares and squeezing through gaps need engine support.) | Half damage at half HP, no healing; shares squares; squeezes without penalty. |
| `tail_spike_regrowth` | manticore | Regrows all 24 tail spikes after a long rest. | uses_left['tail_spike'] reset on long rest. |
| `telepathic_bond` | homunculus | Telepathic bond with its master (no telepathy channel). | No telepathy channel. |
| `trampling_charge` | warhorse | Warhorse: like Pounce with the hooves attack (STR DC 14 or prone, then another hooves attack as a bonus action against a prone target). | Same as pounce. |
| `transparent` | gelatinous_cube | Gelatinous cube: DC 15 Perception to spot while it has neither moved nor attacked. | Like false_appearance, DC 15. |
| `tree_stride` | dryad | Teleports between trees (no tree objects on the maps). | No tree objects. |
| `turning_defiance` | ghast | The ghast has advantage on saves against effects that turn undead (ghouls nearby: see docs, needs an engine hook). | Ghast and ghouls within 30 ft (`ally_save_mods`). |
| `two_heads` | death_dog, ettin | Advantage on Perception checks and on saves against being blinded, charmed, deafened, frightened, stunned or knocked unconscious. | Same tag limitation; advantage vs blinded/deafened/stunned/unconscious only for tagged saves. |
| `undead_fortitude` | ogre_zombie, zombie | CON save DC 5 + the damage taken (unless radiant or a critical hit) to drop to 1 HP instead of 0. | Replaces the engine exemplar: DC = 5 + total damage taken (the exemplar used only the overflow). |
| `vampire_weaknesses` | vampire_spawn | Harmed by running water (ends turn on a water square) and sunlight (start of turn); disadvantage on attack rolls and ability checks in sunlight.  Forbiddance and the stake rule are not modelled. | Sunlight and running-water damage; forbiddance and stake rule not modelled. Water squares count as running water. |
| `variable_illumination` | will_o_wisp | Will-o'-wisp: bright light in a 5- to 20-ft radius (default 10 ft), dim light for the same distance beyond.  The light goes out while the wisp is invisible (see its Invisibility action). | Fixed 10 ft bright + 10 ft dim; off while invisible. |
| `wakeful` | ettin | Ettin heads sleep separately (sleep is not modelled per head). | No per-head sleep. |
| `water_breathing` | reef_shark | Breathes only water (not modelled: no suffocation outside water). | idem |
| `water_form` | water_elemental | Can enter a hostile creature's space (needs engine support); see also 'freeze'. | Shares squares with hostile creatures (engine secondary occupancy). |
| `water_susceptibility` | fire_elemental | Takes cold damage for every 5 ft it moves in water (water squares). | 1 cold damage per water square entered (no gallons). |
| `web_sense` | ettercap, giant_spider, giant_wolf_spider | Knows the location of creatures touching the same web (informational). | Informational. |
| `web_walker` | ettercap, giant_spider, giant_wolf_spider, phase_spider | Ignores movement restrictions caused by webs: cannot be restrained while standing in a web. | Cannot be restrained in a web; web difficult terrain still applies (request 10). |

## Remaining limitations (v1.1)

* Squeezing is modelled on the grid (a Large creature in a 1-wide corridor); amorphous, air_form and swarm are exempt. Gaps narrower than a square (under doors, keyholes) are not modelled.
* running_leap / standing_leap: only long jumps over pits are automatic (STR-based, Jump spell); the monsters' own jump distances are not read.
* tree stride, telepathy, sound/mimicry, planes: no engine model.
* Web burning and the Giant Killer miss-trigger wait for engine call sites (see docs/dev/v11_requests.md).

## Special actions

| id | monsters | implementation | deviations |
|---|---|---|---|
| `aggressive` | orc | Bonus action: move up to its speed toward a hostile creature it can see (the extra movement is not added to the turn's movement). | Extra movement is not added to the turn's movement pool. |
| `consume_life` | will_o_wisp | Bonus action: a living creature at 0 HP within 5 ft makes a CON save (DC 10) or dies; the wisp regains 3d6 HP. | Targets must be downed (0 HP) living creatures within 5 ft. |
| `corrode_metal` | rust_monster | Action: touch a nonmagical metal item of a creature within 5 ft: DEX save (DC 11) or armor / shield loses 1 AC (destroyed at the limit) or a held weapon corrodes (-1 damage, cumulative). | Targets creatures only (objects: no effect modelled). Order: armor, shield, weapon. |
| `create_specter` | wraith | Action: a humanoid that died violently within the last ``dead_for_rounds`` rounds, within 10 ft, rises as a specter under the wraith's control. | Corpses are removed from play but remembered; target selection by eid. |
| `cunning_action` | spy | Bonus action Dash, Disengage or Hide: Disengage when a hostile is adjacent, else Hide if it can, else Dash. | One ability; the option is chosen automatically (disengage > hide > dash). |
| `darkness_aura` | darkmantle | Action (1/day): a 15-ft radius of magical darkness centred on (and following) the darkmantle. | Zone mh_darkness (does not block line of sight). |
| `divine_eminence` | priest | Bonus action: spend a spell slot; melee weapon hits deal +3d6 radiant (+1d6 per slot level above 1st) until the end of the turn. | Spends the lowest available slot. |
| `engulf` | gelatinous_cube | Action: the cube moves up to its speed (extra movement) and may enter Large-or-smaller creatures' spaces; each creature it enters makes a DEX save: failure = engulfed (3d6 acid, grappled + restrained, 6d6 acid at the start of its turns, STR check to escape, moves with the cube), success = pushed 5 ft. One Large or four Medium or smaller creatures. | Capacity rule enforced. |
| `ethereal_jaunt` | phase_spider | Phase spider bonus action: shift to / from the Ethereal Plane (same rules as etherealness). | Same. |
| `etherealness` | ghost | Action: enter / leave the Ethereal Plane.  While ethereal the creature is unseen (invisible), can't be hurt by anything but force damage and can't attack (attacking returns it to the Material Plane). | Invisible + force-only vulnerability; no wall-walking. |
| `grapple_attack` | vampire_spawn | Action: a melee attack roll (``to_hit``) that grapples instead of dealing damage (escape DC ``escape_dc``); counts as the first attack of the monster's Multiattack. | Counts as the first Multiattack attack. |
| `heart_sight` | sprite | Action, touch: learn the target's emotional state; a failed CHA save (DC 10) also reveals its alignment.  Informational: recorded in the event log and in ``mh_heart``. | Informational (event log + fstate). |
| `illusory_appearance` | green_hag | Action: toggle an illusion that makes the hag look like another humanoid (appearance only; statistics unchanged). | Toggle; appearance only. |
| `invisibility` | imp, quasit, sprite, will_o_wisp | Action: become invisible (with carried equipment) until it attacks or its concentration ends (see ``end_on``). | Concentration based on end_on; the wisp's data deviates from the SRD (see below). |
| `invisible_passage` | green_hag | Green hag: invisible until she attacks or casts a spell or her concentration ends (same rules). | Same as invisibility. |
| `leadership` | knight | Bonus action (1/short rest): for 1 minute, nonhostile creatures within 30 ft that can hear the knight add 1d4 to attack rolls and saving throws (one die at a time); ends if the knight is incapacitated. | Allies within 30 ft and in sight (and the knight) get +1d4 to attacks/saves for 10 rounds. |
| `nimble_escape` | goblin | Goblin Nimble Escape: Disengage or Hide as a bonus action (alternates by what helps: hide if possible). |  |
| `parry` | bandit_captain, gladiator, knight, noble | Reaction only: offered through the ON_ATTACKED decision by the 'reaction_parry' trait; never usable as an action. | Reaction only (reaction_parry trait): offered when +AC would turn the hit into a miss. |
| `perception_check` |  | Legendary action: a Wisdom (Perception) check that reveals hidden creatures and traps like the Search action. | Same as Search. |
| `possession` | ghost | Action (Recharge 6): a humanoid within 5 ft makes a CHA save (DC 13) or is possessed: it changes sides (the ghost's faction, scripted control), the ghost leaves the map and returns when the possession ends (0 HP, expiry, or the bonus action granted to the possessed body).  Immune for 24 hours after succeeding or after the possession ends. | Host switches side and becomes scripted-controlled; ghost leaves the map. |
| `rampage` | giant_hyena, gnoll | Bonus action after reducing a creature to 0 HP with a melee attack: move up to half speed and bite. | Moves half speed toward the nearest conscious hostile, then bites if adjacent. |
| `read_thoughts` | doppelganger | Action: read the surface thoughts of a creature within 60 ft (concentration): advantage on Insight, Deception, Intimidation and Persuasion checks against it. | Concentration; advantage on four social skills vs the target. |
| `shriek` | shrieker | Reaction only (shrieker): see the 'reaction_shriek' trait.  Invoking it directly alerts hostiles within earshot. | Reaction only: runs from the shrieker's own turn/round hooks. |
| `swallow` | giant_frog, giant_toad | Action: a bite attack against a creature (``max_size`` or smaller) the monster is grappling; on a hit the target is swallowed (grapple ends; blinded, restrained, total cover, acid at the start of each round, escapes when the monster dies). One swallowed creature at a time. | Victim stays on the map next to the swallower (teleported along); total cover from outsiders. |
| `teleport` | blink_dog | Bonus action (Recharge 4-6): teleport up to 40 ft to an unoccupied space it can see. | Falls back to a free neighbouring square. |

## Effects

| id | monsters | implementation | deviations |
|---|---|---|---|
| `aging` | ghost | Ghost's Horrifying Visage: the target ages 1d4 x 10 years (non-undead only); undead lose the fear too. | Years stored in fstate['mh_aged']; undead are unaffected; 24 h immunity not tracked. |
| `attach_smother` | darkmantle | Darkmantle: attached to the target (grapple + blinded in the data); the darkmantle has advantage against it. | No recurring damage (SRD darkmantle has none). |
| `blood_drain_attach` | stirge | Stirge: attached; at the start of each round the target loses ``dice`` HP; the stirge detaches after draining 10 HP.  The bearer may spend an action to pull it off (granted ability). | Drain at round start; detach action granted to the bearer. |
| `burning` | fire_elemental | engine / spell engineer (`items`) | not registered here |
| `corrode_armor` | black_pudding, gray_ooze | Nonmagical metal armor (or shield, ``slot='shield'``) worn by the bearer takes a permanent, cumulative -1 penalty to the AC it offers; armor reduced to AC 10 / a shield reduced to +0 is destroyed.  The penalties are kept on the creature (``mh_armor_pen``) and applied by this effect's AC bonus. | Armor/shield penalties kept in fstate['mh_armor_pen']. |
| `dreadful_glare` | mummy | Frightened (from the action's condition); paralyzed too if the save failed by 5 or more.  Lasts until the end of the source's next turn (duration 1 round). | 24-hour immunity not tracked. |
| `engulfed` | water_elemental | Engulfed by the source (water elemental / gelatinous cube): ``damage_start_of_turn`` at the start of each of its turns while the grapple lasts; ``cannot_breathe`` is informational. | Damage once per round (round start). |
| `flung` | air_elemental | Air elemental whirlwind: thrown up to ``distance_ft`` in a random direction and knocked prone; 3d6 bludgeoning if it hits an obstacle (a creature in the way makes a DEX save or takes the same damage and falls prone). | Obstacle damage 3d6; creature in the way saves. |
| `luring_song` | harpy | Harpy: the charmed, incapacitated target walks toward the harpy at the start of each of its turns and repeats the WIS save at the end of each turn (success: free of the song).  Only humanoids and giants are affected. | Bonus-action upkeep not required; walks the full speed toward the harpy. |
| `lycanthropy` | wererat, werewolf | Curse of lycanthropy (humanoids only).  A marker: the transformation happens at the next full moon, which the scenario layer may query with ``creature_has_effect(c, 'lycanthropy')``. | Marker curse (transformation not simulated). |
| `max_hp_reduction` | specter, vampire_spawn, wight, wraith | HP maximum reduced by the damage taken (``equals_damage``) or by the necrotic damage (``equals_necrotic``) until a long rest; the creature dies at 0.  ``heals_self``: the attacker regains that many HP. | Amount read back from DAMAGE events; restored on long rest. |
| `mummy_rot` | mummy | Curse: can't regain HP; max HP -3d6 for every 24 h (each long rest stands for 24 h; dies at 0); nothing is lost at the hit; ended by Remove Curse. | Time model: one tick per long rest. |
| `petrification` | cockatrice | The bearer is restrained and 'begins to turn to stone'; at the end of its next turn it repeats the save: success ends the effect, failure petrifies it (petrified condition, no expiry). | Two-stage; first end-of-turn save skipped when applied on the bearer's own turn. |
| `poison_paralysis_at_zero` | giant_centipede, giant_spider, giant_wasp, giant_wolf_spider +1 | If the poison damage dropped the target to 0 HP it is stable but poisoned and paralyzed for the duration. | Data tags the effect `always`. |
| `push` | bronze_dragon_wyrmling | Repulsion breath: pushed ``distance_ft`` away from the source (instant). | Instant. |
| `sleep_poison` | drow, homunculus, pseudodragon, sprite | Poisoned for the duration (not stacked on the POISONED the attack data already applies); unconscious as well if the save failed by 5 or more (wakes on damage). | Failure margin read from the SAVE event. |
| `slowing_breath` | copper_dragon_wyrmling | Copper wyrmling: speed halved and no reactions; repeats the CON save at the end of each of its turns.  (The 'action or bonus action, one attack' limit is not enforced: documented deviation.) | 'One action or bonus action, one attack' limit is not enforced. |
| `smother` | rug_of_smothering | Rug of smothering: while grappled the target is restrained, blinded and suffocating and takes ``damage`` at the start of each of its turns. | Damage at the start of each of the target's turns. |
| `strength_drain` | shadow | STR score reduced by ``dice`` until a short or long rest; dies at 0. | Restored on short/long rest; humanoid-to-shadow rising not modelled. |
| `wasting_disease` | death_dog, otyugh | Disease: each long rest (standing in for 24 hours) the bearer repeats the CON save or loses 1d10 max HP; ends when cured (effect removed). | A long rest stands in for 24 hours. |
| `weakened` | gold_dragon_wyrmling | Gold wyrmling weakening breath: disadvantage on STR-based attack rolls, checks and saves; weapon attacks deal half damage.  Repeats the STR save at the end of each of its turns. | STR-based = non-finesse melee. |
