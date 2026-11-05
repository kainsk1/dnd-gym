# Spells (SRD 5.1)

Generated from `emberdeep/data/spells.py`. Range 0 = self, 5 = touch. Conc = concentration. Hook marks bespoke engine logic.


## Cantrips

| Spell | School | Cast | Range | Area | Conc | Classes | Mechanics | Hook |
|---|---|---|---|---|---|---|---|---|
| Acid Splash | conjuration | A | 60 | - |  | sor,wiz | Hurl acid at a creature, or two creatures within 5 ft of each other; DEX save or 1d6 acid. |  |
| Chill Touch | necromancy | A | 120 | - |  | war,wiz | Ranged spell attack, 1d8 necrotic; target cannot regain HP until your next turn, and an undead target has disadvantage on attacks against you. | chill_touch |
| Dancing Lights | evocation | A | 120 | - | Y | bar,sor,wiz | Create up to four torch-sized lights (dim, 10 ft) that you may move 60 ft as a bonus action. | dancing_lights |
| Eldritch Blast | evocation | A | 120 | - |  | war | Fire beams of force, each a ranged spell attack for 1d10 force; 1 beam, 2 at level 5, 3 at 11, 4 at 17. | eldritch_blast |
| Fire Bolt | evocation | A | 120 | - |  | sor,wiz | Ranged spell attack for 1d10 fire; ignites flammable objects. |  |
| Guidance | divination | A | 5 | - | Y | cle,dru | Touched willing creature adds 1d4 to one ability check within a minute. |  |
| Light | evocation | A | 5 | - |  | bar,cle,sor,wiz | Touched object sheds bright light in 20 ft and dim light 20 ft further for an hour; a hostile holder may DEX-save to negate. | light |
| Mage Hand | conjuration | A | 30 | - |  | bar,sor,war,wiz | A spectral hand within 30 ft can use an object (interact) at range, carrying up to 10 lb, for a minute. | mage_hand |
| Minor Illusion | illusion | A | 30 | - |  | bar,sor,war,wiz | Create a sound or an image of an object in a 5 ft cube for a minute that can block sight and provide a hiding place. | minor_illusion |
| Poison Spray | conjuration | A | 10 | - |  | dru,sor,war,wiz | A creature within 10 ft makes a CON save or takes 1d12 poison. |  |
| Produce Flame | conjuration | A | 0 | - |  | dru | A flame in your hand lights 10 ft bright and 10 ft dim for 10 minutes; hurl it as a ranged spell attack for 1d8 fire. | produce_flame |
| Ray of Frost | evocation | A | 60 | - |  | sor,wiz | Ranged spell attack for 1d8 cold and the target's speed is reduced by 10 ft until the start of your next turn. |  |
| Resistance | abjuration | A | 5 | - | Y | cle,dru | Touched willing creature adds 1d4 to one saving throw within a minute. |  |
| Sacred Flame | evocation | A | 60 | - |  | cle | A target makes a DEX save (cover gives no bonus) or takes 1d8 radiant. |  |
| Shillelagh | transmutation | BA | 5 | - |  | dru | Your club or quarterstaff uses your spellcasting ability and deals d8 magical bludgeoning for a minute. | shillelagh |
| Shocking Grasp | evocation | A | 5 | - |  | sor,wiz | Melee spell attack (advantage vs metal armor) for 1d8 lightning; target cannot take reactions until its next turn. | shocking_grasp |
| Spare The Dying | necromancy | A | 5 | - |  | cle | A dying creature within touch becomes stable. | spare_the_dying |
| Thaumaturgy | transmutation | A | 30 | - |  | cle | A minor wonder: booming voice, tremors, or flames, giving advantage on Intimidation for a minute. | thaumaturgy |
| True Strike | divination | A | 30 | - | Y | bar,sor,war,wiz | You gain advantage on your first attack roll against the target on your next turn. |  |
| Vicious Mockery | enchantment | A | 60 | - |  | bar | A WIS save or 1d4 psychic and disadvantage on its next attack roll before the end of its next turn. |  |

## Level 1

| Spell | School | Cast | Range | Area | Conc | Classes | Mechanics | Hook |
|---|---|---|---|---|---|---|---|---|
| Animal Friendship | enchantment | A | 30 | - |  | bar,dru,ran | A beast with INT 3 or less makes a WIS save or is charmed for 24 hours. | animal_friendship |
| Bane | enchantment | A | 30 | - | Y | bar,cle | Up to three creatures make CHA saves or subtract 1d4 from attack rolls and saves for a minute. |  |
| Bless | enchantment | A | 30 | - | Y | cle,pal | Up to three creatures add 1d4 to attack rolls and saves for a minute. |  |
| Burning Hands | evocation | A | 0 | cone 15 |  | sor,wiz | A 15 ft cone of flame; DEX save for half of 3d6 fire. |  |
| Charm Person | enchantment | A | 30 | - |  | bar,dru,war,sor,wiz | A humanoid makes a WIS save (advantage if you are fighting it) or is charmed by you for an hour. | charm_person |
| Color Spray | illusion | A | 0 | cone 15 |  | sor,wiz | A 15 ft cone blinds creatures in order of lowest current HP, using a pool of 6d10 HP. | color_spray |
| Command | enchantment | A | 60 | - |  | cle,pal | A creature that can hear and understand you makes a WIS save or obeys a one-word command on its next turn. | command |
| Cure Wounds | evocation | A | 5 | - |  | bar,cle,dru,pal,ran | A touched creature regains 1d8 + spellcasting modifier HP. |  |
| Detect Magic | divination | A | 0 | - | Y | bar,cle,dru,pal,ran,sor,wiz | For 10 minutes you sense magic within 30 ft and may see its aura and school. | detect_magic |
| Divine Favor | evocation | BA | 0 | - | Y | pal | Your weapon attacks deal an extra 1d4 radiant damage for a minute. |  |
| Entangle | conjuration | A | 90 | cube 20 | Y | dru | Grasping plants fill a 20 ft square; creatures there make a STR save or are restrained (escape by STR check). |  |
| Expeditious Retreat | transmutation | BA | 0 | - | Y | sor,wiz | You can take the Dash action as a bonus action each turn for ten minutes. |  |
| Faerie Fire | evocation | A | 60 | cube 20 | Y | bar,dru | Creatures in a 20 ft cube make DEX saves or are outlined, giving attackers advantage against them and revealing invisible ones. |  |
| False Life | necromancy | A | 0 | - |  | sor,wiz | You gain 1d4+4 temporary HP for an hour (+5 per slot level above 1st). | false_life |
| Feather Fall | transmutation | R | 60 | - |  | bar,sor,wiz | As a reaction up to five falling creatures descend slowly and take no falling damage. | feather_fall |
| Fog Cloud | conjuration | A | 120 | sphere 20 | Y | dru,ran,sor,wiz | A 20 ft radius sphere of fog heavily obscures its area for an hour. |  |
| Goodberry | transmutation | A | 0 | - |  | dru,ran | Creates ten berries, each healing 1 HP as an action to eat and feeding for a day. | goodberry |
| Grease | conjuration | A | 60 | cube 10 |  | wiz | A 10 ft square becomes slick; creatures there make a DEX save or fall prone. |  |
| Guiding Bolt | evocation | A | 120 | - |  | cle | Ranged spell attack for 4d6 radiant; the next attack against the target before your next turn ends has advantage. |  |
| Healing Word | evocation | BA | 60 | - |  | bar,cle,dru | As a bonus action, a creature within 60 ft regains 1d4 + spellcasting modifier HP. |  |
| Hellish Rebuke | evocation | R | 60 | - |  | war | Reaction when damaged by a creature within 60 ft: it makes a DEX save, taking 2d10 fire (half on success). | hellish_rebuke |
| Heroism | enchantment | A | 5 | - | Y | bar,pal | A willing creature is immune to fear and gains temporary HP equal to your spellcasting modifier at the start of each of its turns. |  |
| Tasha's Hideous Laughter | enchantment | A | 30 | - | Y | bar,wiz | A creature with INT 5+ makes a WIS save or falls prone, laughing and incapacitated; it repeats the save each turn and when damaged. |  |
| Hunter's Mark | divination | BA | 90 | - | Y | ran | Mark a creature: your weapon attacks deal an extra 1d6 damage to it and you track it; move the mark as a bonus action when it dies. | hunters_mark |
| Inflict Wounds | necromancy | A | 5 | - |  | cle | Melee spell attack for 3d10 necrotic. |  |
| Jump | transmutation | A | 5 | - |  | dru,ran,sor,wiz | A creature's jump distance triples for a minute. |  |
| Longstrider | transmutation | A | 5 | - |  | bar,dru,ran,wiz | A creature's speed increases by 10 ft for an hour. |  |
| Mage Armor | abjuration | A | 5 | - |  | sor,wiz | An unarmored willing creature has base AC 13 + DEX modifier for eight hours. |  |
| Magic Missile | evocation | A | 120 | - |  | sor,wiz | Three darts each automatically deal 1d4+1 force, divided among targets as you choose; one more dart per slot level above 1st. |  |
| Protection from Evil and Good | abjuration | A | 5 | - | Y | cle,pal,war,wiz | Aberrations, celestials, elementals, fey, fiends, and undead have disadvantage attacking the creature, which cannot be charmed, frightened, or possessed by them. |  |
| Sanctuary | abjuration | BA | 30 | - |  | cle | A creature is warded: anyone targeting it with an attack or harmful spell must first make a WIS save or choose a new target. | sanctuary |
| Shield | abjuration | R | 0 | - |  | sor,wiz | Reaction when hit or targeted by magic missile: +5 AC until your next turn start, and magic missile cannot harm you. | shield |
| Shield of Faith | abjuration | BA | 60 | - | Y | cle,pal | A shimmering field gives a creature +2 AC for ten minutes. |  |
| Sleep | enchantment | A | 90 | sphere 20 |  | bar,sor,wiz | Roll 5d8 HP: creatures in a 20 ft radius fall unconscious in ascending HP order up to that total (undead and charm-immune are unaffected). | sleep |
| Thunderwave | evocation | A | 0 | cube 15 |  | bar,dru,sor,wiz | A 15 ft cube blast; CON save for half of 2d8 thunder, and creatures that fail are pushed 10 ft. | thunderwave |

## Level 2

| Spell | School | Cast | Range | Area | Conc | Classes | Mechanics | Hook |
|---|---|---|---|---|---|---|---|---|
| Melf's Acid Arrow | evocation | A | 90 | - |  | wiz | Ranged spell attack: 4d4 acid on a hit plus 2d4 at the end of its next turn; a miss deals half the initial damage and nothing later. | acid_arrow |
| Aid | abjuration | A | 30 | - |  | cle,pal | Up to three creatures raise their HP maximum and current HP by 5 for eight hours (+5 per slot above 2nd). | aid |
| Barkskin | transmutation | A | 5 | - | Y | dru,ran | A willing creature's AC cannot be lower than 16 for an hour. |  |
| Blindness/Deafness | necromancy | A | 30 | - |  | bar,cle,sor,wiz | Choose blindness or deafness: a CON save or the target has that condition for a minute (repeat save each turn). | blindness_deafness |
| Blur | illusion | A | 0 | - | Y | sor,wiz | Attack rolls against you have disadvantage for a minute (not vs. truesight/blindsight). |  |
| Calm Emotions | enchantment | A | 60 | sphere 20 | Y | bar,cle | Humanoids in a 20 ft radius make CHA saves (willing may fail); choose to suppress charm/fear or make them indifferent to enemies. | calm_emotions |
| Darkness | evocation | A | 60 | sphere 15 | Y | sor,war,wiz | A 15 ft radius sphere of magical darkness for ten minutes that darkvision cannot penetrate and blocks nonmagical light. |  |
| Darkvision | transmutation | A | 5 | - |  | dru,ran,sor,wiz | A willing creature gains darkvision 60 ft for eight hours. |  |
| Enhance Ability | transmutation | A | 5 | - | Y | bar,cle,dru,sor | Choose an ability: advantage on its checks plus a side benefit (e.g. +2d6 carry, 1d6 fall reduction, temporary HP) for an hour. | enhance_ability |
| Enlarge/Reduce | transmutation | A | 30 | - | Y | sor,wiz | Enlarge: size up, advantage on STR checks/saves, +1d4 weapon damage; Reduce: size down, disadvantage on STR, -1d4 weapon damage. Unwilling creatures make a CON save. | enlarge_reduce |
| Find Traps | divination | A | 0 | - |  | cle,dru,ran | You sense whether traps (and their general nature) lie within 120 ft in line of sight. | find_traps |
| Flame Blade | evocation | BA | 0 | - | Y | dru | A fiery blade appears in your hand; melee spell attack for 3d6 fire (+1d6 per two slot levels above 2nd). | flame_blade |
| Flaming Sphere | conjuration | A | 60 | - | Y | dru,wiz | A 5 ft fire sphere rolls where you direct as a bonus action; creatures ending adjacent or rammed make a DEX save for half of 2d6 fire. | flaming_sphere |
| Gust of Wind | evocation | A | 0 | line 60 | Y | dru,sor,wiz | A 60 ft by 10 ft line of wind: STR save or pushed 15 ft; difficult terrain against it, extinguishes flames. | gust_of_wind |
| Heat Metal | transmutation | A | 60 | - | Y | bar,dru | Heat a metal object worn or held: 2d8 fire now and each turn as a bonus action; the holder must drop it or take disadvantage on attacks and checks. | heat_metal |
| Hold Person | enchantment | A | 60 | - | Y | bar,cle,dru,sor,war,wiz | A humanoid makes a WIS save or is paralyzed for a minute, repeating the save each turn. |  |
| Invisibility | illusion | A | 5 | - | Y | bar,sor,war,wiz | A creature becomes invisible for an hour; it ends on attacking or casting. |  |
| Knock | transmutation | A | 60 | - |  | bar,sor,wiz | An object (door, chest, shackles) is unlocked or unstuck; a loud knock is heard 300 ft away. | knock |
| Lesser Restoration | abjuration | A | 5 | - |  | bar,cle,dru,pal,ran | End one disease or one condition on a touched creature: blinded, deafened, paralyzed, or poisoned. | lesser_restoration |
| Levitate | transmutation | A | 60 | - | Y | sor,wiz | A creature or loose object rises up to 20 ft and hovers for ten minutes (unwilling: CON save). |  |
| Magic Weapon | transmutation | BA | 5 | - | Y | pal,wiz | A nonmagical weapon becomes +1 (+2 at slot 4, +3 at slot 6) for an hour. | magic_weapon |
| Mirror Image | illusion | A | 0 | - |  | sor,war,wiz | Three illusory duplicates make attacks against you often target a duplicate instead. |  |
| Misty Step | conjuration | BA | 0 | - |  | sor,war,wiz | As a bonus action, teleport up to 30 ft to an unoccupied space you can see. | misty_step |
| Moonbeam | evocation | A | 120 | cylinder 5 | Y | dru | A 5 ft radius column of light: CON save for half of 2d10 radiant on entering or starting a turn inside (shapechangers have disadvantage); you can move it. |  |
| Pass Without Trace | abjuration | A | 0 | emanation 30 | Y | dru,ran | You and creatures within 30 ft gain +10 to Stealth checks and cannot be tracked for an hour. |  |
| Prayer of Healing | evocation | 1m+ | 30 | - |  | cle | Over ten minutes, up to six creatures regain 2d8 + spellcasting modifier HP (not usable in combat). |  |
| Protection from Poison | abjuration | A | 5 | - |  | cle,dru,pal,ran | End one poison affecting a creature; for an hour it has advantage on saves against poison and resistance to poison damage. | protection_from_poison |
| Ray of Enfeeblement | necromancy | A | 60 | - | Y | war,wiz | Ranged spell attack: the target deals half damage with STR-based weapon attacks, repeating a CON save at the end of each turn. |  |
| Scorching Ray | evocation | A | 120 | - |  | sor,wiz | Three rays, each a separate ranged spell attack for 2d6 fire, targeted as you choose; one more ray per slot level above 2nd. |  |
| See Invisibility | divination | A | 0 | - |  | bar,sor,wiz | You see invisible creatures and objects and into the Ethereal Plane for an hour. |  |
| Shatter | evocation | A | 60 | sphere 10 |  | bar,sor,war,wiz | A 10 ft radius burst of noise: CON save for half of 3d8 thunder (inorganic creatures have disadvantage). |  |
| Silence | illusion | A | 120 | sphere 20 | Y | bar,cle,ran | A 20 ft radius of silence for ten minutes: creatures inside are deafened and immune to thunder, and verbal spells cannot be cast. |  |
| Spider Climb | transmutation | A | 5 | - | Y | sor,war,wiz | A creature can climb walls and ceilings with its hands free and gains a climb speed equal to its walk speed for ten minutes. |  |
| Spike Growth | transmutation | A | 150 | sphere 20 | Y | dru,ran | A 20 ft radius becomes hidden spikes: difficult terrain, and 2d4 piercing for every 5 ft a creature moves through it. |  |
| Spiritual Weapon | evocation | BA | 60 | - |  | cle | A floating weapon you can attack with as a bonus action (melee spell attack, 1d8 + spellcasting mod force, within 20 ft move) for a minute. | spiritual_weapon |
| Suggestion | enchantment | A | 30 | - | Y | bar,sor,wiz | A creature makes a WIS save or pursues a reasonable course of action you describe for up to eight hours. | suggestion |
| Warding Bond | abjuration | A | 5 | - |  | cle,pal | A willing creature gains +1 AC, +1 saves and resistance to all damage for an hour while you take the same damage it takes. | warding_bond |
| Web | conjuration | A | 60 | cube 20 | Y | sor,wiz | A 20 ft cube of webs: difficult terrain; creatures make a DEX save or are restrained (STR check to escape); flammable. |  |

## Level 3

| Spell | School | Cast | Range | Area | Conc | Classes | Mechanics | Hook |
|---|---|---|---|---|---|---|---|---|
| Animate Dead | necromancy | 1m+ | 10 | - |  | cle,wiz | Over a minute, raise a skeleton or zombie from bones or a corpse as your minion for 24 hours. | animate_dead |
| Beacon of Hope | abjuration | A | 30 | - | Y | cle | Chosen creatures have advantage on WIS and death saves and regain maximum HP from any healing for a minute. |  |
| Bestow Curse | necromancy | A | 5 | - | Y | bar,cle,wiz | A touched creature makes a WIS save or is cursed (choose: disadvantage on one ability, disadvantage attacking you, WIS save to act, or +1d8 necrotic from you). | bestow_curse |
| Blink | transmutation | A | 0 | - |  | sor,wiz | At the end of each of your turns roll a d20; on 11+ you vanish to the Ethereal Plane until the start of your next turn. | blink |
| Call Lightning | conjuration | A | 120 | sphere 5 | Y | dru | A storm cloud lets you call down a lightning bolt each turn: DEX save for half of 3d10 lightning in a 5 ft radius. | call_lightning |
| Conjure Animals | conjuration | A | 60 | - | Y | dru,ran | Summon fey spirits as beasts: one CR 2, two CR 1, four CR 1/2, or eight CR 1/4 that obey you for an hour. | conjure_animals |
| Counterspell | abjuration | R | 60 | - |  | sor,war,wiz | Reaction: interrupt a spell being cast within 60 ft; slots of 3rd or lower auto-counter, higher need an ability check. | counterspell |
| Daylight | evocation | A | 60 | sphere 60 |  | cle,dru,pal,ran,sor | A 60 ft radius sphere of bright light (dim 60 ft more) for an hour that dispels darkness of lower level. | daylight |
| Dispel Magic | abjuration | A | 120 | - |  | bar,cle,dru,pal,sor,war,wiz | End spells of 3rd level or lower on a target; higher-level ones need an ability check against DC 10 + spell level. | dispel_magic |
| Fear | illusion | A | 0 | cone 30 | Y | bar,sor,war,wiz | A 30 ft cone: WIS save or drop what you hold and flee frightened for a minute (repeat save each turn when out of sight). | fear |
| Fireball | evocation | A | 150 | sphere 20 |  | sor,wiz | A 20 ft radius explosion: DEX save for half of 8d6 fire. |  |
| Fly | transmutation | A | 5 | - | Y | sor,war,wiz | A touched creature gains a 60 ft fly speed for ten minutes. |  |
| Gaseous Form | transmutation | A | 5 | - | Y | sor,war,wiz | A willing creature becomes a misty cloud with fly 10 ft, resistance to nonmagical physical damage, advantage on STR/DEX/CON saves, and cannot attack or cast. |  |
| Haste | transmutation | A | 30 | - | Y | sor,wiz | A willing creature gains +2 AC, advantage on DEX saves, doubled speed, and an extra action (attack once, dash, disengage, hide, use object); when it ends it is lethargic for a turn. | haste |
| Hypnotic Pattern | illusion | A | 120 | cube 30 | Y | bar,war,sor,wiz | A 30 ft cube of colors: WIS save or be charmed and incapacitated with speed 0 for a minute; damage or a shake ends it. | hypnotic_pattern |
| Lightning Bolt | evocation | A | 0 | line 100 |  | sor,wiz | A 100 ft long, 5 ft wide line: DEX save for half of 8d6 lightning. |  |
| Mass Healing Word | evocation | BA | 60 | - |  | cle | As a bonus action, up to six creatures within 60 ft regain 1d4 + spellcasting modifier HP. |  |
| Protection from Energy | abjuration | A | 5 | - | Y | cle,dru,ran,sor,wiz | Choose acid, cold, fire, lightning, or thunder: the target has resistance to it for an hour. | protection_from_energy |
| Remove Curse | abjuration | A | 5 | - |  | cle,pal,war,wiz | End all curses affecting a creature or object, including breaking attunement to a cursed item. | remove_curse |
| Revivify | necromancy | A | 5 | - |  | cle,pal | Return a creature that died within the last minute to life with 1 HP (300 gp diamonds consumed). | revivify |
| Sleet Storm | conjuration | A | 150 | cylinder 40 | Y | dru,sor,wiz | A 40 ft radius, 20 ft high storm: difficult terrain, extinguished flames, heavily obscured; creatures make DEX saves or fall prone, and concentration checks are at disadvantage. | sleet_storm |
| Slow | transmutation | A | 120 | - | Y | sor,wiz | Up to six creatures make WIS saves or are slowed for a minute (half speed, -2 AC and DEX saves, one action or bonus action, no reactions); they repeat the save each turn. |  |
| Spirit Guardians | conjuration | A | 0 | emanation 15 | Y | cle | Spirits fill a 15 ft radius around you: enemies are slowed to half speed there and make a WIS save for half of 3d8 radiant or necrotic each turn. |  |
| Stinking Cloud | conjuration | A | 90 | sphere 20 | Y | bar,sor,wiz | A 20 ft radius gas cloud heavily obscures its area; creatures starting or entering make a CON save or lose their action poisoned. |  |
| Vampiric Touch | necromancy | A | 5 | - | Y | war,wiz | Melee spell attack for 3d6 necrotic and you regain half the damage dealt; reattack each turn as an action for the duration. | vampiric_touch |
| Wind Wall | evocation | A | 120 | line 50 | Y | dru,ran | A 50 ft by 15 ft wall of wind: STR save or 3d8 bludgeoning (half on success); blocks gas, small flying creatures and nonmagical arrows. | wind_wall |

## Level 4

| Spell | School | Cast | Range | Area | Conc | Classes | Mechanics | Hook |
|---|---|---|---|---|---|---|---|---|
| Banishment | abjuration | A | 60 | - | Y | cle,pal,sor,war,wiz | A creature makes a CHA save or is banished to a harmless demiplane (removed from play) for a minute; extraplanar creatures are gone for good if concentration lasts. | banishment |
| Greater Invisibility | illusion | A | 5 | - | Y | bar,sor,wiz | A creature is invisible for a minute and stays invisible even when it attacks or casts. |  |
| Guardian of Faith | conjuration | A | 30 | sphere 10 |  | cle | A spectral guardian for eight hours damages hostile creatures that come within 10 ft: DEX save for half of 20 radiant, vanishing after 60 total damage. | guardian_of_faith |
| Ice Storm | evocation | A | 300 | cylinder 20 |  | dru,sor,wiz | A 20 ft radius hail: DEX save for half of 2d8 bludgeoning plus 4d6 cold; the area becomes difficult terrain until your next turn. |  |

## Level 5

| Spell | School | Cast | Range | Area | Conc | Classes | Mechanics | Hook |
|---|---|---|---|---|---|---|---|---|
| Cone of Cold | evocation | A | 0 | cone 60 |  | sor,wiz | A 60 ft cone of cold: CON save for half of 8d8 cold. |  |

## Manifests

Hooks (63): acid_arrow, aid, animal_friendship, animate_dead, banishment, bestow_curse, blindness_deafness, blink, call_lightning, calm_emotions, charm_person, chill_touch, color_spray, command, conjure_animals, counterspell, dancing_lights, daylight, detect_magic, dispel_magic, eldritch_blast, enhance_ability, enlarge_reduce, false_life, fear, feather_fall, find_traps, flame_blade, flaming_sphere, goodberry, guardian_of_faith, gust_of_wind, haste, heat_metal, hellish_rebuke, hunters_mark, hypnotic_pattern, knock, lesser_restoration, light, mage_hand, magic_weapon, minor_illusion, misty_step, produce_flame, protection_from_energy, protection_from_poison, remove_curse, revivify, sanctuary, shield, shillelagh, shocking_grasp, sleep, sleet_storm, spare_the_dying, spiritual_weapon, suggestion, thaumaturgy, thunderwave, vampiric_touch, warding_bond, wind_wall


Effects (62): acid_arrow, aid, bane, banished, barkskin, beacon_of_hope, bestow_curse, bless, blink, blur, calm_emotions, chill_touch, color_spray, commanded, darkvision, divine_favor, enhance_ability, enlarge, expeditious_retreat, faerie_fire, feather_fall, flame_blade, fly, gaseous_form, guidance_die, guiding_bolt, haste, heat_metal, heroism, hideous_laughter, hold_person, hunters_mark, hunters_mark_tracker, hypnotized, invisibility, jump, levitate, longstrider, mage_armor, magic_weapon, mirror_image, no_reactions, pass_without_trace, protection_from_energy, protection_from_evil_and_good, protection_from_poison, ray_of_enfeeblement, resistance_die, sanctuary, see_invisibility, shield, shield_of_faith, shillelagh, sleep, slow, speed_penalty, spider_climb, suggested, thaumaturgy, true_strike, vicious_mockery, warding_bond


Zones (19): call_lightning, dancing_lights, darkness, daylight, entangle, flaming_sphere, fog_cloud, grease, guardian_of_faith, ice_storm, light, moonbeam, silence, sleet_storm, spike_growth, spirit_guardians, stinking_cloud, web, wind_wall
