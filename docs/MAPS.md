# Emberdeep maps

Everything in `emberdeep.scenarios` that produces a `MapData`: the ASCII parser (`ascii_map`), the procedural
generators (`mapgen`), the decorator (`decorate`) and the hand-designed library (`library/maps/*.txt`).
All of it is deterministic in its `seed`.

## Conventions

* `terrain`/`material` are `uint8[H, W]` indexed `[y, x]`; `Terrain.FLOOR` underlies doors, chests, altars, traps and
  spawn markers (a marker cell is a plain floor cell).
* **Cell centre = `(x + 0.5, y + 0.5)`** for `LightData` and `Decor` positions (continuous coordinates). `ObjectData`,
  `DoorData`, markers and regions use integer cell coordinates.
* `DoorData.orientation`: `"h"` when walls flank the door left/right (a north-south passage), `"v"` when walls are
  above/below (an east-west passage). Locked doors have `lock_dc` 12-16 (`L` in ASCII; `door_locked:15` forces a DC).
* Objects: `chest` params `{locked_dc, gold (dice string), items [item ids / key ids], trapped ("" | "poison_needle" |
  "fire_glyph"), vault}`; `trap` params `{trap: spike_pit|poison_dart|fire_glyph|net, dc: 13, hidden: True}`;
  `loot` params `{gold}`; `altar` (name = label, e.g. "Fountain of Life"); `lever`; `exit` (also `Terrain.EXIT`).
* Lights: torch 20/20 ft warm; brazier 30/30 ft; lava 0/15-20 ft red-orange; crystal 0/10 ft coloured; mushrooms
  `kind="magic"` 0/10 ft teal; candles 5/10 ft; campfire `kind="fire"` 20/20 ft.
* Multi-cell props (sarcophagi) are a single `Decor` at the footprint centre with `scale` = length in squares; the
  terrain cells under it are `LOW_WALL`. Free-standing cosmetics never sit on blocking terrain; wall-mounted decor
  (`banner`, `torch_sconce`, `crystal`, `tree`, `chains`, `grate`, ...) sits on `WALL` cells only.

## ASCII legend

| char | meaning | terrain / extras |
|---|---|---|
| `#` | wall | `WALL` (material from `MapSpec.wall_material`) |
| `.` `,` | floor / alt floor | `FLOOR` (`floor_material` with natural variation: moss, cobble near walls, dirt patches / `alt_floor_material`) |
| space | void | `VOID` (rows are right-padded with void) |
| `+` `/` `L` | closed / open / locked door | `FLOOR` + `DoorData`, orientation auto-detected |
| `=` | lava | `LAVA` + `Material.LAVA`, dim red-orange `LightData` per cluster |
| `~` `w` | shallow / deep water | `WATER_SHALLOW` / `WATER_DEEP` |
| `:` `"` | rubble / brush | `DIFFICULT` (`RUBBLE`) / `BRUSH` |
| `O` `b` `i` | pit / bridge / ice | `PIT` (`CHASM`) / `BRIDGE` (planks) / `ICE` |
| `P` `n` `S` | pillar / low wall or crates / statue | `PILLAR` + `pillar_ornate`, `LOW_WALL` + crate/barrel (themed), `STATUE` + `statue` |
| `T` | wall torch | `WALL` + `torch_sconce` decor + 20/20 ft torch light on the adjacent floor side |
| `t` | standing brazier | `LOW_WALL` + `brazier` decor + 30/30 ft light |
| `E` | exit | `EXIT` + `ObjectData("exit")` + `stairs_down`; auto region `exit` |
| `C` `A` `$` `l` | chest / altar / loot pile / lever | floor + `ObjectData` |
| `^` | hidden trap | floor + `ObjectData("trap")`, trap kind chosen by seed |
| `0-9`, lowercase except `b i l n t w` | spawn marker | floor; `spawn_markers[char]` |

`MapSpec.legend` extends/overrides: `{char: token}` with tokens `wall floor floor_alt void door door_open door_locked[:dc]
lava water deep_water rubble brush pit pillar low_wall[:decor] statue bridge ice exit trap chest chest_locked[:dc] torch
brazier altar[:name] loot lever spawn:<marker name>`.

### `.txt` header format (`ascii_map.load_map_file`)

Lines starting with `;` are comments unless they match a header `;key: value`:

```
;theme: crypt                      MapSpec.theme (dungeon crypt cave castle forest sewer temple lair arena tavern)
;floor: STONE_FLOOR                floor material (core.types.Material name)
;alt_floor: COBBLE                 material of ','
;wall: STONE_WALL                  wall material
;ambient: dark                     dark | dim | bright
;density: 1.0                      decor density multiplier (0 = structural props only)
;legend X: spawn:boss              extend the legend
;light: x y bright_ft dim_ft kind [r g b]   extra light at cell (x, y)
;region treasure: 10,4 11,4        cells (repeatable, accumulates)
;region vault: rect 10 4 14 8      inclusive rectangle
```

Hand-made maps get only their explicit lights (`T`, `t`, `;light:`, lava) plus structural props and a light cosmetic
pass; procedural maps also get sconces, braziers, crystals and mushrooms from `decorate`.

## Generators

`mapgen.generate(generator, params, seed)`; unknown parameters raise `ValueError`. CLI:
`python -m emberdeep.scenarios.mapgen --gen cave --seed 3 [--w 48 --h 36] [--param key=value] [--png out.png]`.
Every map guarantees 4-connected reachability of all walkable cells from the entrance, regions `entrance`, `exit`,
`treasure`, `boss`, `room_<id>` (and `corridors` where meaningful), spawn markers `entrance` (>= 8 cells),
`room_<id>`, `exit`, `boss`, `treasure`, and calls `decorate`.

| generator | defaults | notes |
|---|---|---|
| `bsp_dungeon` | `width=48`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `rooms=0` (auto = width*height/190), `min_room=5`, `door_prob=0.75`, `lock_prob=0.2`, `key_prob=0.5`, `loop_prob=0.35`, `wide_corridor_prob=0.25`, `pillar_prob=0.7`, `boss=True`, `shrine=True`, `barracks=True` | BSP rooms (rect / chamfered / oval / cross / L-shaped), L/Z corridors 1-2 wide, loops, doors at 1-wide room entrances (some locked 12-16 with keys held in chests reachable without them), entrance / hall / vault (locked, 2 chests) / boss (exit stairs, throne) / shrine (altar + candles) / barracks (beds). |
| `cave` | `width=48`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `fill=0.46`, `iterations=5`, `chambers=8`, `water=0.5`, `chasm=0.4`, `rubble=0.5`, `mushrooms=1.0`, `lava=0.0`, `min_pocket=30` | cellular automata, tunnelled connectivity, shallow/deep pools, chasms (bridged where they cut the cave), rubble, mushroom glades, optional lava vents, graph-Voronoi chambers; vault = far dead-end grotto, boss chamber holds the exit. |
| `crypt` | `width=49`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `tiers=0`, `ossuary_prob=0.5`, `niches=True`, `lock_prob=1.0` | left-right symmetric: antechamber, nave with pillars and statue niches, tiers of sarcophagus chambers (alternating ossuary bone floors), ring corridors (flank routes) behind locked doors, two vaults, boss tomb. Width is rounded up to odd. |
| `arena` | `width=44`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `gates=6`, `pillars=8`, `braziers=4`, `lock_prob=0.5`, `cover=6` | oval pit with spectator rail, pillars, braziers, cover, hidden spike traps and `gates` gate chambers behind doors; exit in the opposite gate, prize chests in a third. |
| `forest_clearing` | `width=48`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `trees=0.5`, `brush=0.5`, `stream=True`, `ruins=True`, `clearings=3`, `glades=3` | noise tree walls (HEDGE) with clearings, a dirt road west to east, side paths to glades (one with a campfire), a meandering stream (continuous deep channel) with one plank bridge on the road and a ford, brush, ruins with a vault. |
| `sewer` | `width=48`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `tunnels_x=2`, `tunnels_y=2`, `chambers=3`, `bridges=2`, `narrow_prob=0.35` | grid of tunnels (walkway / shallow / deep channel / shallow / walkway), junction plazas with a pillar, bridges, collapsed rubble, grates at the outfalls, chambers with pools (boss, locked vault). |
| `temple` | `width=41`, `height=36`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `chapels=2`, `pews=True`, `lock_prob=1.0` | left-right symmetric: narthex, nave with columns and pews, carpeted aisle, apse with altar dais and communion rail, side chapels, two sacristy vaults (one keyed). Width rounded up to odd. |
| `lair` | `width=44`, `height=32`, `chest_count=3`, `trap_count=3`, `decor_density=1.0`, `alcoves=3`, `hoard_radius=4`, `pillars=7`, `lava=1`, `loot_piles=8` | large organic chamber with stalagmites and lava, hoard (loot piles + vault chests) at the east end, winding entrance tunnel, kobold alcoves (some on loops) and an exit alcove behind the hoard. |

Common parameters: `width`, `height`, `chest_count`, `trap_count`, `decor_density`, `theme` (override the visual
theme), `floor_material`, `wall_material`. BSP: `rooms`, `min_room`, `door_prob`, `lock_prob`, `key_prob`, `loop_prob`,
`wide_corridor_prob`, `pillar_prob`, `boss`, `shrine`, `barracks`. Cave: `fill`, `iterations`, `chambers`, `water`,
`chasm`, `rubble`, `mushrooms`, `lava`, `min_pocket`. Lair: `alcoves`, `hoard_radius`, `pillars`, `lava`,
`loot_piles`. Others as listed in the table.

## Library maps

### `goblin_ambush` - cooperation baseline (36x24)

goblin_ambush - forest road ambush (cooperation baseline), 36x24.
Party travels west -> east along the dirt road; the exit stairs 'E' lie at the far east end.
Markers: 0-5 party start (west end of the road)
a  goblins hidden in the north-west brush (3 cells)   c  north-east brush (3 cells)
d  goblins hidden in the south-west brush (3 cells)   e  south-east brush (3 cells)
f  goblin archers behind the tree clumps east of the choke (2 cells)
g  goblin boss / leader on the road past the choke (1 cell)
Features: 2-wide choke at x=20..24 flanked by boulders 'S'; fallen logs 'n' = half cover;
'^' goblin pit traps on the road; 'C' abandoned wagon chest in the south brush; '$' goblin stash NE;
small pond in the south-west corner. Regions: road, north_brush, south_brush.

```
####################################
####################################
############.#########.#############
###########......#......############
#..#####.."""""""........""#######.#
#......."""""a"""""...."""""c""".$.#
#......""""a"""a""""..""""c""""""..#
#.......#"""""""""###..""""""c""...#
#......###""""""".###...."""""S###.#
#.......#nn.........#####......###.#
#...........,,,,,,,S#####......f...#
#,03,,,,,,,,,,,^,,,,,,,,,,,,,,,,,,,#
#,14,,,,,,,,,,,,,,,,,,,,,,,^,,,g,,E#
#,25,,,,,,,,,.......#####S,,,,nn,,,#
#.........nnn.......#####......f##.#
#.....###..""".S...###...."""..###.#
#.....###"""""""".#####""""""""###.#
#......"""d"""d""".###"""""e"""""..#
#.~~~~~"""""d"""""...."""e"""e"""..#
#~~~#~~~"""""""""..C...""""#######.#
#########.."""..#........###########
##########.###########..############
#######################..###########
####################################
```

### `dragon_hoard` - stag hunt (44x32)

dragon_hoard - stag hunt: wyrmling + huge hoard in the deep chamber vs. small kobold-guarded alcove chests, 44x32.
Markers: 0-5 party start (entrance cavern, west)   d dragon wyrmling at the hoard (1 cell)
k kobolds guarding the north alcove chest (3)   j kobolds guarding the south alcove chest (3)
m kobold sentries in the junction cavern (2)    g kobold guards at the chamber mouth (2)
Objects: 'E' escape stairs behind the party; 3 hoard chests + 8 loot piles '$' in the east; 2 small alcove chests
(north (18,5), south (17,27)); hidden traps '^' in tunnels; lava pools flank the chamber; the alcoves also
connect to the chamber through 1-wide flanking tunnels (loops). Regions: hoard, alcove_north, alcove_south, dragon_chamber.

```
############################################
############################################
############################################
###################.########################
################.......#####################
###############...C..k..#########.....######
###############....k..........##.......#####
###############...k.....#####...........####
################.......#######.......:...###
###################....#######..===......###
####################...######...===.......##
####################.^.######......P......##
####.....###########...#####...P.........C.#
##.........########.....####..:.........$..#
##.........#.....#.......###........:.$....#
#...03................m......g.........$.$.#
#.E.14.........^................^.P....d..$#
#...25.......###......m......g.........$..C#
##.........#######.......###........:....$.#
##.........########.....####..:.........$..#
####.....###########...#####...P.........C.#
####################.^.######......P......##
####################...######...===.......##
###################....#######..===......###
################.......#######.......:...###
###############...j.....#####...........####
###############....j..........##.......#####
###############..C...j..#########.....######
################.......#####################
###################.########################
############################################
############################################
```

### `cursed_coffer` - prisoner's dilemma over loot (30x22)

cursed_coffer - prisoner's dilemma over loot: one coffer, two entrances, guardians, 30x22.
Markers: 0-3 west group (behind the LOCKED door at (6,10), lock DC rolled 12-16)
4-7 east group (behind the unlocked door at (23,11))
g guardians west of the dais (2)   h guardians east of the dais (2)
Objects: 'C' the cursed coffer on the dais at (14,10); hidden traps '^' at (14,9) and (14,11);
statues 'S' at the dais corners; pillars 'P' and crates 'n' give cover; braziers 't' light the hall,
the antechambers are dark. 'E' escape stairs at the back of each antechamber. Regions: dais, vault.

```
##############################
##############################
##########T########T##########
#########............#########
########..............########
#######.....t.n..t.....#######
###T###..P..........P..###T###
#.....#................#.....#
#.....#.n...S,,,S......#.....#
#.01..#.n..g,,^,,.h....#.....#
#.23..L...P.,,C,,......#..45.#
#.....#....g,,^,,.hP...+..67.#
#.....#.....S,,,S....n.#.....#
#E....#..............n.#....E#
#.....#................#.....#
#######..P..........P..#######
#######.....t..n.t.....#######
########..............########
#########............#########
##########T########T##########
##############################
##############################
```

### `rival_delvers` - two-party relic race (truce vs fight) (49x32)

rival_delvers - two parties race for a central relic (truce vs fight), mirror-symmetric, 49x32.
Markers: 0-3 west party start (start room, west)   4-7 east party start (mirror)
a north-west monster room (4)  c north-east monster room (4)  d south-west monster room (4)  e south-east (4)
r relic guardians inside the relic room (2)
Objects: 'A' the relic altar at (24,15) with hidden traps '^' at (24,14),(24,16); 'E' escape stairs at each start room.
Each hub reaches the relic through a LOCKED door (picking the lock is quick but noisy) or through its monster room
(guarded chest 'C') and an unlocked door on the relic room's north/south side. The two north (and south) monster
rooms are linked by a shared corridor, so the parties can meet before reaching the relic. Regions: relic_room, start_west, start_east.

```
#################################################
#################################################
###############T#################T###############
##########...........#######...........##########
##########.C.........#######.........C.##########
##########..Pa..a.P..#######..P.c..cP..##########
##########...........+.....+...........##########
##########....a..a...###.###...c..c....##########
##########....n......###.###......n....##########
##########...........###.###...........##########
#####T#########+########+########+#########T#####
##........#####.####.........####.#####........##
##......n.####....##.t.....t.##....####.n......##
##........###T....T#..P...P..#T....T###........##
##.02.....####....##....^....##....####.....64.##
##.13.....+........L..r.A.r..L........+.....75.##
##........####....##....^....##....####........##
##........####....##..P...P..##....####........##
##........####....##.t.....t.##....####........##
##......n.#####.####.........####.#####.n......##
##E.......#####.########+########.#####.......E##
#####T#########.########.########.#########T#####
###############+########.########+###############
##########...........###.###...........##########
##########......n....###.###....n......##########
##########....d..d...###.###...e..e....##########
##########...........+.....+...........##########
##########..Pd..d.P..#######..P.e..eP..##########
##########.C.........#######.........C.##########
##########...........#######...........##########
###############T#################T###############
#################################################
```

### `fountain_of_life` - common-pool resource (37x29)

fountain_of_life - common-pool resource: a regenerating healing fountain under monster waves, 37x29.
Markers: 0-5 party start on the plinth around the fountain
a north wave room (5)   c south wave room (5)   d east wave room (5)   e west wave room (5)
Objects: 'A' the Fountain of Life at (18,14) (MapData altar object named 'Fountain of Life'; charges/regen set by the scenario).
Four 3-wide entrances; statues 'S' flank each mouth (3/4 cover); pillars 'P' ring; braziers 't' on the diagonals.
No exit: waves are survived. Regions: fountain, chamber.

```
#####################################
##############..aaaaa..##############
##############.........##############
#################...#################
#################...#################
#################...#################
##############..S...S..##############
#############...........#############
############t.....P.....t############
###########...............###########
##########....P.......P....##########
#...######.................######...#
#e..######S....n.012.n....S######..d#
#e.......#.......,,,.......#.......d#
#e..........P....,A,....P..........d#
#e.......#.......,,,.......#.......d#
#e..######S....n.345.n....S######..d#
#...######.................######...#
##########....P.......P....##########
###########...............###########
############t.....P.....t############
#############...........#############
##############..S...S..##############
#################...#################
#################...#################
#################...#################
##############.........##############
##############..ccccc..##############
#####################################
```

### `shrine_of_wards` - threshold public good (33x26)

shrine_of_wards - threshold public good: donate gold at the shrine for a party-wide ward while waves attack, 33x26.
Markers: 0-5 party start in the narthex (south)
a north-west chapel (3)  c north-east chapel (3)  d south-west chapel (3)  e south-east chapel (3)
f waves emerging inside the apse next to the shrine (2)
Objects: 'A' donation shrine at (16,3); 'C' chests in the two upper chapels; pews 'n' and columns 'P' fill the nave
(cover everywhere), the communion rail at y=5 leaves a 3-wide gap. Chapel doors open onto the nave and the narthex,
so waves can flank. Regions: shrine, nave.

```
#################################
#################################
############.f,,,,,f.############
############.S,,A,,S.############
############t.,,,,,.t############
############.nn...nn.############
###########...........###########
###C......#...........#......C###
###.......T..P.....P..T.......###
###.a.....#...nn.nn...#.....c.###
###..a....+...........+....c..###
###.a.....#..Pnn.nnP..#.....c.###
###.......#...........#.......###
######+####...nn.nn...####+######
###.......T..P.....P..T.......###
###.d.....#...nn.nn...#.....e.###
###.d.....+...........+.....e.###
###..d....#..Pnn.nnP..#....e..###
###.......#...........#.......###
######+#########+#########+######
####.........................####
####.t.....................t.####
####.........012345..........####
####.........................####
####.........................####
#################################
```

### `hold_the_bridge` - volunteer's dilemma (40x24)

hold_the_bridge - volunteer's dilemma: one narrow bridge over a chasm, treasure on the far side, exit behind the party, 40x24.
Markers: 0-5 party start (west bank, near the exit)
a north enemy squad (3)  c south enemy squad (3)  d vanguard at the bridge head (2)  f flankers (2)
g archers on the rubble ledges (2)  k enemy captain (1)
Objects: 'E' exit stairs at (2,11); 2 chests + 5 loot piles '$' in the east alcove (x>=36); hidden traps '^' at the east bridge head.
The bridge 'b' is 1 square wide and 10 long (y=11, x=15..24); the chasm 'O' spans the full height. Pillars/crates
at either bridge head give cover, braziers 't' light both heads. Regions: bridge, west_bank, east_bank, treasure.

```
########################################
########################################
########################################
#####..........OOOOOOOOOO..........#####
####...........OOOOOOOOOO...........####
###............OOOOOOOOOO........:...###
#......P.......OOOOOOOOOO....g.........#
#..............OOOOOOOOOO.....n..aaa...#
#..........P.t.OOOOOOOOOO.t.P.n....S...#
#.............:OOOOOOOOOO:.......f...$.#
#...03......nn.OOOOOOOOOO..n...d....$.C#
#.E.14.........bbbbbbbbbb.^^.....k...$.#
#...25......nn.OOOOOOOOOO..n...d....$.C#
#.............:OOOOOOOOOO:.......f...$.#
#..........P.t.OOOOOOOOOO.t.P......S...#
#..............OOOOOOOOOO.....n..ccc...#
#..............OOOOOOOOOO.....n........#
#......P.......OOOOOOOOOO....g.........#
###............OOOOOOOOOO.........:..###
####...........OOOOOOOOOO...........####
#####..........OOOOOOOOOO..........#####
########################################
########################################
########################################
```

### `gladiator_pit` - free-for-all with alliances (32x32)

gladiator_pit - free-for-all with alliances: six gladiators, one prize, 32x32.
Markers: 0-5 one start cell per gate chamber, each behind a closed door in the spectator rail:
0 north-west (11,3)  1 north-east (20,3)  2 south-west (11,28)  3 south-east (20,28)  4 west (2,16)  5 east (29,15)
Objects: 'C' prize chest at the centre (15,15) on a rubble dais with 4 statues 'S'; hidden spike traps '^' ring the dais;
8 pillars 'P' at radius 6, 6 braziers 't' at radius 8, low walls 'n' = half cover; the rail 'n' around the pit gives half cover.
No exit. Regions: arena, center.

```
################################
################################
##########...######...##########
##########.0.######.1.##########
##########...nnnnnn...##########
##########n+nn....nn+n##########
#########nn..........nn#########
########nn.....t......nn########
#######n................n#######
######nn.......n........nn######
#####nn......P....P......nn#####
#####n..t..........n...t..n#####
#####n.........^..........n#####
####nn....P..S....S..P....nn####
####n......................n...#
#...n....n..^.:C:..........+.5.#
#.4.+..........::..^..n....n...#
#...n......................n####
####nn....P..S....S..P....nn####
#####n..........^.........n#####
#####n..t...n..........t..n#####
#####nn......P....P......nn#####
######nn........n.......nn######
#######n................n#######
########nn......t.....nn########
#########nn..........nn#########
##########n+nn....nn+n##########
##########...nnnnnn...##########
##########.2.######.3.##########
##########...######...##########
################################
################################
```

### `caravan_escort` - protect the NPC / bribe the bandits (56x22)

caravan_escort - protect the caravan along a canyon road; bandits can be fought or bribed, 56x22.
Markers: 0-3 escort party  v caravan wagon / NPC cells (2)  E exit stairs at the east end (54,11)
a north ledge 1 (3)  c south ledge 1 (3)  d north ledge 2 (3)  e south ledge 2 (3)  f north ledge 3 (3)  g south ledge 3 (3)
h bandit toll line at the east end of the gully bridge (3)  k bandit leader beyond the second choke (1)
Features: canyon with two 3-wide chokes (x=20..22, x=38..40) and a dry gully 'O' crossed by a 3-wide bridge (x=30..32);
ledges overlook the road behind a low rail 'n' (half cover) with ramps ':'; boulders 'S' give cover on the road;
loot: chests 'C' on south ledge 2 and north ledge 3, loot piles '$' on north ledge 2. Regions: road, gully.

```
########################################################
########################################################
######............####........############..........####
######...aa.......####..dd....############...ff.C...####
######.....a......####....d.$.############.....f....####
######............####.....$..############..........####
######nnnnn::nnnnn####nnn::nnn############nnnn::nnnn####
#......###..........###.......OOO########......#######.#
#.....#####.........###.......OOO########.......#####..#
#......###,,,,S,....###....S..OOO,,,,,###,,.S.......S..#
#,02,:,,,,,,,,,,,,,,,,.....,,,bbb,h,,,,,,,,,,,,,,,,,,,,#
#,,vv,,,,,,,,:,,,,,,,,,,^,,,:,bbb,h,,,,,,,^,k,,,,,,,,,E#
#,13,,,.S......,,,,,,,,,,,,,,,bbb,h:S.....,:,,,,,,,,,,,#
#.................S.###,,,,S..OOO.....###.......S......#
#............###....#######...OOO.....###...#####......#
#...........#####...########..OOO.....###..#######.....#
############nnnn::nnnn###########nnnn::nnn####nnn::nnn##
############..........###########.........####........##
############...c.c....###########...e.e...####...g.g..##
############....c.....###########...Ce....####....g...##
############..........###########.........####........##
########################################################
```
