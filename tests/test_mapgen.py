"""Tests for ASCII parsing, decoration and procedural map generation."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from emberdeep.core.types import BLOCKS_MOVE, DoorState, Material, Terrain
from emberdeep.scenarios import ascii_map, decorate, mapgen
from emberdeep.scenarios.mapdata import MapData
from emberdeep.scenarios.spec import MapSpec

T = Terrain
LIB = Path(ascii_map.__file__).parent / "library" / "maps"
BLOCK = sorted(int(t) for t in BLOCKS_MOVE)
SEEDS = range(30)


def parse(rows, seed=0, **kw) -> MapData:
    return ascii_map.parse(MapSpec(ascii=list(rows), **kw), seed)


def reachable_from(md: MapData, start) -> np.ndarray:
    walk = ~np.isin(md.terrain, BLOCK)
    seen = np.zeros_like(walk)
    seen[start[1], start[0]] = True
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < md.width and 0 <= ny < md.height and walk[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    return seen


def ring_runs(md: MapData, x: int, y: int) -> int:
    walk = ~np.isin(md.terrain, BLOCK)
    ring = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    vals = [bool(0 <= x + dx < md.width and 0 <= y + dy < md.height and walk[y + dy, x + dx]) for dx, dy in ring]
    return sum((not vals[i]) and vals[(i + 1) % 8] for i in range(8))


# ----------------------------------------------------------------------------------------------
# ASCII parsing
# ----------------------------------------------------------------------------------------------

EXPECTED_TERRAIN = {
    "#": T.WALL, ".": T.FLOOR, ",": T.FLOOR, " ": T.VOID, "=": T.LAVA, "~": T.WATER_SHALLOW, "w": T.WATER_DEEP,
    ":": T.DIFFICULT, '"': T.BRUSH, "O": T.PIT, "P": T.PILLAR, "n": T.LOW_WALL, "S": T.STATUE, "b": T.BRIDGE,
    "i": T.ICE, "E": T.EXIT, "^": T.FLOOR, "C": T.FLOOR, "T": T.WALL, "t": T.LOW_WALL, "A": T.FLOOR, "$": T.FLOOR,
    "l": T.FLOOR, "+": T.FLOOR, "/": T.FLOOR, "L": T.FLOOR,
}


@pytest.mark.parametrize("ch", sorted(EXPECTED_TERRAIN))
def test_legend_terrain(ch):
    md = parse(["#####", f"#.{ch}.#", "#####"])
    assert Terrain(md.terrain[1, 2]) == EXPECTED_TERRAIN[ch]
    assert md.terrain.dtype == np.uint8 and md.material.dtype == np.uint8
    assert md.terrain.shape == md.material.shape == (3, 5)


def test_legend_materials():
    md = parse(["#######", "#.,=~w#", "#:O\"bi#", "#######"])
    m = lambda x, y: Material(md.material[y, x])  # noqa: E731
    assert m(0, 0) == Material.STONE_WALL
    assert m(2, 1) == Material.COBBLE            # alt floor default
    assert m(3, 1) == Material.LAVA and m(4, 1) == Material.WATER and m(5, 1) == Material.DEEP_WATER
    assert m(1, 2) == Material.RUBBLE and m(2, 2) == Material.CHASM and m(5, 2) == Material.ICE
    assert m(4, 2) == Material.WOOD_PLANKS


def test_spawn_markers_and_padding_and_comments():
    md = parse(["; a comment", "#####", "#01a#", "#.z", "#####"])
    assert md.width == 5 and md.height == 4           # comment skipped, ragged rows padded
    assert md.spawn_markers["0"] == [(1, 1)] and md.spawn_markers["a"] == [(3, 1)] and md.spawn_markers["z"] == [(2, 2)]
    assert all(md.terrain[y, x] == T.FLOOR for cells in md.spawn_markers.values() for x, y in cells)
    assert md.terrain[2, 4] == T.VOID                  # padding is void
    # non-markers: b, i, l, n, t, w keep their terrain meaning
    assert "b" not in md.spawn_markers and "n" not in md.spawn_markers


def test_unknown_char_raises():
    with pytest.raises(ValueError):
        parse(["#@#"])


def test_door_orientation():
    # corridor running north-south through a horizontal wall: walls left/right -> 'h'
    ns = parse(["#.#", "#+#", "#.#"])
    assert ns.doors[0].orientation == "h"
    # corridor running east-west through a vertical wall: walls above/below -> 'v'
    ew = parse(["###", ".+.", "###"])
    assert ew.doors[0].orientation == "v"
    # a door in a wall line between two rooms (walls above/below along the vertical wall)
    room = parse(["#####", "#.#.#", "#.+.#", "#.#.#", "#####"])
    assert room.doors[0].orientation == "v"
    wide = parse(["#######", "#.....#", "###+###", "#.....#", "#######"])
    assert wide.doors[0].orientation == "h"


def test_door_states_and_lock_dc():
    rows = ["#######", "#.+/L.#", "#######"]
    md = parse(rows, seed=3)
    st = {d.x: d for d in md.doors}
    assert st[2].state == DoorState.CLOSED and st[3].state == DoorState.OPEN and st[4].state == DoorState.LOCKED
    assert 12 <= st[4].lock_dc <= 16 and st[2].lock_dc == 0
    assert parse(rows, seed=3).doors[2].lock_dc == st[4].lock_dc       # deterministic
    custom = parse(["#####", "#.L.#", "#####"], legend={"L": "door_locked:15"})
    assert custom.doors[0].lock_dc == 15


def test_torch_and_brazier():
    md = parse(["#####", "#.T.#", "#...#", "#.t.#", "#####"])
    # torch at (2,1) is not on a wall edge: still wall + sconce
    assert md.terrain[1, 2] == T.WALL
    assert any(d.kind == "torch_sconce" and int(d.x) == 2 and int(d.y) == 1 for d in md.decor)
    torch = [l for l in md.lights if l.kind == "torch"]
    assert len(torch) == 1 and torch[0].bright_ft == 20 and torch[0].dim_ft == 20
    # the light sits on the floor side of the wall square
    md2 = parse(["###T###", "#.....#", "#######"])
    lt = [l for l in md2.lights if l.kind == "torch"][0]
    assert 3.0 <= lt.x <= 4.0 and 1.0 <= lt.y < 2.0 and md2.terrain[int(lt.y), int(lt.x)] == T.FLOOR
    br = [l for l in md.lights if l.kind == "brazier"]
    assert len(br) == 1 and br[0].bright_ft > 20
    assert md.terrain[3, 2] == T.LOW_WALL and any(d.kind == "brazier" for d in md.decor)


def test_structural_decor_matches_terrain():
    md = parse(["#######", "#.P.nS#", "#.....#", "#######"])
    kinds = {(int(d.x), int(d.y)): d.kind for d in md.decor}
    assert kinds[(2, 1)] == "pillar_ornate"
    assert kinds[(4, 1)] in ("crate", "barrel")
    assert kinds[(5, 1)] == "statue"


def test_objects_and_exit_and_traps():
    rows = ["#########", "#C.A.$.l#", "#.^.^.^.E", "#########"]
    md = parse(rows, seed=5)
    kinds = sorted(o.kind for o in md.objects)
    assert kinds == sorted(["chest", "altar", "loot", "lever", "trap", "trap", "trap", "exit"])
    assert md.terrain[2, 8] == T.EXIT and "exit" in md.regions
    for o in md.objects:
        if o.kind == "trap":
            assert o.params["trap"] in ("spike_pit", "poison_dart", "fire_glyph", "net")
            assert o.params["dc"] == 13 and o.params["hidden"] is True
    again = parse(rows, seed=5)
    assert [o.params for o in again.objects] == [o.params for o in md.objects]
    kinds_by_seed = {tuple(o.params["trap"] for o in parse(rows, seed=s).objects if o.kind == "trap") for s in range(12)}
    assert len(kinds_by_seed) > 1                      # trap kinds vary with the seed


def test_lava_light_and_lights_merge():
    spec = MapSpec(ascii=["#######", "#.===.#", "#.===.#", "#######"], lights=[{"x": 1, "y": 1, "bright_ft": 10, "dim_ft": 5, "kind": "magic"}],
                   regions={"zone": [(1, 1), (2, 1)]}, ambient_light="dim", theme="cave")
    md = ascii_map.parse(spec, 0)
    assert any(l.kind == "lava" and l.bright_ft == 0 and l.color[0] > l.color[2] for l in md.lights)
    assert any(l.kind == "magic" and l.x == 1.5 for l in md.lights)
    assert md.regions["zone"] == [(1, 1), (2, 1)] and md.ambient == "dim" and md.theme == "cave"


def test_legend_extension():
    spec = MapSpec(ascii=["#####", "#X.Y#", "#####"], legend={"X": "spawn:boss", "Y": "chest_locked:15"})
    md = ascii_map.parse(spec, 0)
    assert md.spawn_markers["boss"] == [(1, 1)]
    assert md.objects[0].kind == "chest" and md.objects[0].params["locked_dc"] == 15


def test_to_ascii_roundtrip():
    rows = ["###########", "#0..T..=~w#", "#.P.n.S:\"O#", "+..^..C.bE#", "#t.A.$.l.i#", "###########"]
    md = parse(rows, seed=1)
    out = ascii_map.to_ascii(md)
    assert out == rows
    assert np.array_equal(parse(out, seed=1).terrain, md.terrain)


def test_materials_deterministic_and_vary():
    rows = ["#" * 40] + ["#" + "." * 38 + "#" for _ in range(30)] + ["#" * 40]
    a = parse(rows, seed=1, theme="sewer")
    b = parse(rows, seed=1, theme="sewer")
    c = parse(rows, seed=2, theme="sewer")
    assert np.array_equal(a.material, b.material)
    assert not np.array_equal(a.material, c.material)
    assert len(np.unique(a.material[a.terrain == T.FLOOR])) > 1      # moss / cobble variation


def test_load_map_file_header(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("\n".join([
        "; free comment: not a header", ";theme: crypt", ";floor: BONE_FLOOR", ";alt_floor: COBBLE", ";wall: BRICK_WALL",
        ";ambient: dim", ";density: 0.5", ";legend X: spawn:boss", ";light: 2 1 20 10 torch 1 0.5 0.2",
        ";region treasure: 1,1 2,1", ";region zone: rect 1 1 3 2", ";region treasure: 3,1",
        "#####", "#X..#", "#...#", "#####", ""]))
    spec = ascii_map.load_map_file(p)
    assert spec.theme == "crypt" and spec.floor_material == "BONE_FLOOR" and spec.alt_floor_material == "COBBLE"
    assert spec.wall_material == "BRICK_WALL" and spec.ambient_light == "dim" and spec.decor_density == 0.5
    assert spec.legend == {"X": "spawn:boss"} and spec.lights[0]["kind"] == "torch" and spec.lights[0]["color"] == (1.0, 0.5, 0.2)
    assert spec.regions["treasure"] == [(1, 1), (2, 1), (3, 1)] and len(spec.regions["zone"]) == 6
    assert spec.ascii == ["#####", "#X..#", "#...#", "#####"]
    md = ascii_map.parse(spec, 0)
    assert md.spawn_markers["boss"] == [(1, 1)] and md.theme == "crypt"


# ----------------------------------------------------------------------------------------------
# library maps
# ----------------------------------------------------------------------------------------------

LIBRARY = {
    "goblin_ambush": ((36, 24), "012345acdefg", {"road", "north_brush", "south_brush"}),
    "dragon_hoard": ((44, 32), "012345dkjmg", {"hoard", "alcove_north", "alcove_south"}),
    "cursed_coffer": ((30, 22), "01234567gh", {"dais", "vault"}),
    "rival_delvers": ((49, 32), "01234567acder", {"relic_room", "start_west", "start_east"}),
    "fountain_of_life": ((37, 29), "012345acde", {"fountain", "chamber"}),
    "shrine_of_wards": ((33, 26), "012345acdef", {"shrine", "nave"}),
    "hold_the_bridge": ((40, 24), "012345acdfgk", {"bridge", "west_bank", "east_bank", "treasure"}),
    "gladiator_pit": ((32, 32), "012345", {"arena", "center"}),
    "caravan_escort": ((56, 22), "0123vacdefghk", {"road", "gully"}),
}


@pytest.mark.parametrize("name", sorted(LIBRARY))
def test_library_map(name):
    size, markers, regions = LIBRARY[name]
    path = LIB / f"{name}.txt"
    text = path.read_text()
    spec = ascii_map.load_map_file(path)
    md = ascii_map.parse(spec, 0)
    assert (md.width, md.height) == size
    for m in markers:
        assert md.spawn_markers.get(m), f"{name}: marker {m!r} missing"
        # every documented marker is mentioned in the header comments
        assert any(line.startswith(";") and (m in line) for line in text.splitlines()[:30])
    assert regions <= set(md.regions) and all(md.regions[r] for r in regions)
    assert text.startswith(";") and "Markers" in text
    # every walkable cell is reachable from the first party marker
    seen = reachable_from(md, md.spawn_markers["0"][0])
    walk = ~np.isin(md.terrain, BLOCK)
    assert not (walk & ~seen).any(), f"{name}: unreachable floor"
    # every marker is reachable
    for cells in md.spawn_markers.values():
        assert all(seen[y, x] for x, y in cells)
    # border is sealed
    assert np.isin(md.terrain[0, :], BLOCK).all() and np.isin(md.terrain[-1, :], BLOCK).all()
    assert np.isin(md.terrain[:, 0], BLOCK).all() and np.isin(md.terrain[:, -1], BLOCK).all()
    # lights in bounds, regions walkable-ish cells in bounds
    for l in md.lights:
        assert 0 <= l.x < md.width and 0 <= l.y < md.height
    for cells in md.regions.values():
        assert all(0 <= x < md.width and 0 <= y < md.height for x, y in cells)
    _check_decor(md)
    assert ascii_map.to_ascii(md)  # inverse works


def test_library_special_objects():
    md = ascii_map.parse(ascii_map.load_map_file(LIB / "fountain_of_life.txt"), 0)
    fountain = [o for o in md.objects if o.kind == "altar"]
    assert len(fountain) == 1 and fountain[0].name == "Fountain of Life" and (fountain[0].x, fountain[0].y) == (18, 14)
    assert any(d.kind == "fountain" for d in md.decor)
    md = ascii_map.parse(ascii_map.load_map_file(LIB / "hold_the_bridge.txt"), 0)
    assert sum(1 for c in md.regions["bridge"]) == 10
    assert all(md.terrain[y, x] == T.BRIDGE for x, y in md.regions["bridge"])
    md = ascii_map.parse(ascii_map.load_map_file(LIB / "rival_delvers.txt"), 0)
    assert np.array_equal(md.terrain, md.terrain[:, ::-1])                         # mirror symmetric
    assert any(d.state == DoorState.LOCKED for d in md.doors)
    md = ascii_map.parse(ascii_map.load_map_file(LIB / "cursed_coffer.txt"), 0)
    assert sum(1 for o in md.objects if o.kind == "chest") == 1 and len(md.doors) == 2
    md = ascii_map.parse(ascii_map.load_map_file(LIB / "gladiator_pit.txt"), 0)
    assert len(md.doors) == 6 and all(len(md.spawn_markers[str(i)]) == 1 for i in range(6))


# ----------------------------------------------------------------------------------------------
# decor consistency (shared by library + generated maps)
# ----------------------------------------------------------------------------------------------

COSMETIC_FLOOR_ONLY = {"bones", "skull", "blood_stain", "puddle", "glyph", "moss_patch", "rug", "mushrooms", "candles",
                       "cobweb", "rubble", "bush"}
PROP_FOR = {
    T.LOW_WALL: {"crate", "barrel", "brazier", "sarcophagus", "coffin", "fence", "rock", "bookshelf", "table", "weapon_rack",
                 "cage", "anvil", "cauldron", "well", "fountain", "bed"},
    T.STATUE: {"statue"},
    T.PILLAR: {"pillar_ornate", "rock"},
}


def _check_decor(md: MapData) -> None:
    by_cell: dict[tuple[int, int], list[str]] = {}
    for d in md.decor:
        assert 0 <= d.x < md.width and 0 <= d.y < md.height, f"decor {d.kind} out of bounds"
        by_cell.setdefault((int(d.x), int(d.y)), []).append(d.kind)
        t = Terrain(md.terrain[int(d.y), int(d.x)])
        if d.kind in COSMETIC_FLOOR_ONLY:
            assert t not in BLOCKS_MOVE and t not in (T.WATER_DEEP, T.LAVA, T.PIT), f"{d.kind} on {t.name}"
        if t in PROP_FOR and d.kind not in ("torch_sconce",):
            assert d.kind in PROP_FOR[t], f"{d.kind} on {t.name} at {(int(d.x), int(d.y))}"
        if t in (T.WALL, T.VOID):
            assert d.kind in decorate.WALL_MOUNTED, f"{d.kind} on a wall"
    # every connected group of prop-terrain cells (multi-cell props such as sarcophagi are one Decor at the
    # footprint centre) carries at least one matching prop
    for t, kinds_ok in PROP_FOR.items():
        for comp in ascii_map._components(md.terrain == t):
            have = [k for c in comp for k in by_cell.get(c, [])]
            assert any(k in kinds_ok for k in have), f"{t.name} group at {comp[0]} has no matching prop ({have})"
    for l in md.lights:
        assert 0 <= l.x < md.width and 0 <= l.y < md.height


def test_decorate_never_changes_terrain_cosmetically():
    rows = ["#" * 30] + ["#" + "." * 28 + "#" for _ in range(20)] + ["#" * 30]
    md = parse(rows, seed=4, theme="dungeon")
    before = md.terrain.copy()
    rng = np.random.default_rng(0)
    decorate.decorate(md, "crypt", rng, 3.0, lighting=False)
    assert np.array_equal(before, md.terrain)
    assert any(d.kind in ("bones", "skull", "cobweb") for d in md.decor)
    _check_decor(md)


def test_decorate_density_zero_has_no_cosmetics():
    rows = ["#" * 20] + ["#" + "." * 18 + "#" for _ in range(12)] + ["#" * 20]
    md = parse(rows, seed=0, decor_density=0.0)
    assert md.decor == []


def test_decorate_lighting_sconces_and_braziers():
    md = mapgen.generate("bsp_dungeon", None, 2)
    torch = [l for l in md.lights if l.kind == "torch"]
    assert torch
    spacing = min(np.hypot(a.x - b.x, a.y - b.y) for i, a in enumerate(torch) for b in torch[i + 1:])
    assert spacing >= 3.0
    sconces = [d for d in md.decor if d.kind == "torch_sconce"]
    assert all(md.terrain[int(d.y), int(d.x)] == T.WALL for d in sconces)


# ----------------------------------------------------------------------------------------------
# generators
# ----------------------------------------------------------------------------------------------

GENS = sorted(mapgen.GENERATORS)


@pytest.mark.parametrize("gen", GENS)
def test_generator_all_seeds(gen):
    for seed in SEEDS:
        md = mapgen.generate(gen, None, seed)
        h, w = md.height, md.width
        assert md.terrain.shape == md.material.shape == (h, w)
        assert md.terrain.dtype == np.uint8 and md.material.dtype == np.uint8
        assert md.seed == seed
        # connectivity from the entrance
        ent = md.spawn_markers["entrance"]
        assert len(ent) >= 8, f"{gen}/{seed}: entrance markers"
        seen = reachable_from(md, ent[0])
        walk = ~np.isin(md.terrain, BLOCK)
        assert not (walk & ~seen).any(), f"{gen}/{seed}: unreachable cells"
        assert mapgen.check_connectivity(md)
        # border sealed
        assert np.isin(md.terrain[0, :], BLOCK).all() and np.isin(md.terrain[:, 0], BLOCK).all()
        assert np.isin(md.terrain[-1, :], BLOCK).all() and np.isin(md.terrain[:, -1], BLOCK).all()
        # regions promised
        for r in ("entrance", "exit", "treasure", "boss"):
            assert md.regions.get(r), f"{gen}/{seed}: region {r} empty"
        for room in md.rooms:
            assert md.regions[f"room_{room.id}"] == room.cells and room.cells
            assert room.x0 <= min(c[0] for c in room.cells) and room.x1 >= max(c[0] for c in room.cells)
            assert all(not (md.terrain[y, x] in BLOCK) for x, y in room.cells)
            assert all(n != room.id and room.id in md.rooms[n].neighbors for n in room.neighbors)
            assert f"room_{room.id}" in md.spawn_markers or not any(md.terrain[y, x] == T.FLOOR for x, y in room.cells)
        assert md.rooms[0].depth >= 0 and any(r.depth == 0 for r in md.rooms)
        # markers sit on free floor
        occupied = {(o.x, o.y) for o in md.objects} | {(d.x, d.y) for d in md.doors}
        for name, cells in md.spawn_markers.items():
            for x, y in cells:
                assert md.terrain[y, x] == T.FLOOR and (x, y) not in occupied, f"{gen}/{seed}: marker {name}"
        # doors / objects
        for d in md.doors:
            assert md.terrain[d.y, d.x] == T.FLOOR
            assert d.orientation == ascii_map.door_orientation(md.terrain, d.x, d.y)
            if d.state == DoorState.LOCKED:
                assert 12 <= d.lock_dc <= 16
        exits = [o for o in md.objects if o.kind == "exit"]
        assert exits and all(md.terrain[o.y, o.x] == T.EXIT for o in exits)
        for o in md.objects:
            assert o.kind in ("chest", "trap", "exit", "altar", "loot", "lever")
            assert 0 <= o.x < w and 0 <= o.y < h and md.terrain[o.y, o.x] not in BLOCK or o.kind == "exit"
            if o.kind == "trap":
                assert o.params["hidden"] and o.params["trap"] in ascii_map.TRAP_KINDS
        assert sum(1 for o in md.objects if o.kind == "chest") >= 1
        # decor & lights
        _check_decor(md)
        assert md.lights, f"{gen}/{seed}: no lights"
        assert md.ambient in ("dark", "dim", "bright")


@pytest.mark.parametrize("gen", GENS)
def test_generator_deterministic(gen):
    a = mapgen.generate(gen, None, 11)
    b = mapgen.generate(gen, None, 11)
    c = mapgen.generate(gen, None, 12)
    assert np.array_equal(a.terrain, b.terrain) and np.array_equal(a.material, b.material)
    assert a.doors == b.doors and a.objects == b.objects and a.lights == b.lights and a.decor == b.decor
    assert a.spawn_markers == b.spawn_markers and a.regions == b.regions and a.rooms == b.rooms
    assert not np.array_equal(a.terrain, c.terrain) or a.objects != c.objects


@pytest.mark.parametrize("gen", GENS)
def test_generator_params_and_sizes(gen):
    defaults = mapgen.default_params(gen)
    with pytest.raises(ValueError):
        mapgen.generate(gen, {"no_such_param": 1}, 0)
    big = mapgen.generate(gen, {"width": defaults["width"] + 10, "height": defaults["height"] + 8}, 3)
    assert big.width >= defaults["width"] + 10 - 1 and big.height == defaults["height"] + 8
    assert mapgen.check_connectivity(big)
    _check_decor(big)
    small = mapgen.generate(gen, {"decor_density": 0.0, "chest_count": 0, "trap_count": 0}, 5)
    assert not [o for o in small.objects if o.kind == "trap"]
    assert mapgen.check_connectivity(small)


def test_unknown_generator():
    with pytest.raises(ValueError):
        mapgen.generate("nope", None, 0)


def test_bsp_features():
    locked = keyed = chests_with_keys = 0
    for seed in SEEDS:
        md = mapgen.generate("bsp_dungeon", {"lock_prob": 0.6, "key_prob": 1.0}, seed)
        kinds = {r.kind for r in md.rooms}
        assert {"entrance", "boss", "vault"} <= kinds
        for d in md.doors:
            if d.key_id:
                keyed += 1
                assert any(d.key_id in o.params.get("items", []) for o in md.objects if o.kind == "chest")
        locked += sum(d.state == DoorState.LOCKED for d in md.doors)
        chests_with_keys += sum(bool(o.params.get("items")) for o in md.objects if o.kind == "chest")
        vault = [r for r in md.rooms if r.kind == "vault"][0]
        assert any(o.kind == "chest" and (o.x, o.y) in set(vault.cells) for o in md.objects)
        # corridors width 1..2 exist; more than a tree: loops
    assert locked > 10 and keyed > 5 and chests_with_keys >= keyed
    loops = 0
    for seed in range(10):
        md = mapgen.generate("bsp_dungeon", {"loop_prob": 1.0}, seed)
        loops += sum(len(r.neighbors) for r in md.rooms) // 2 - (len(md.rooms) - 1)
    assert loops > 0


def test_lava_vents_and_lights():
    md = mapgen.generate("cave", {"lava": 3, "water": 1.0, "chasm": 1.0}, 7)
    assert (md.terrain == T.LAVA).any() and any(l.kind == "lava" for l in md.lights)
    assert (md.terrain == T.WATER_SHALLOW).any()
    assert mapgen.check_connectivity(md)


def test_crypt_symmetry_and_arena_gates():
    md = mapgen.generate("crypt", None, 1)
    terr = md.terrain.copy()
    for d in md.decor:                      # braziers are placed by decorate at random corners
        if d.kind == "brazier":
            terr[int(d.y), int(d.x)] = T.FLOOR
    assert np.array_equal(terr, terr[:, ::-1])
    md = mapgen.generate("arena", {"gates": 6}, 2)
    assert len([r for r in md.rooms if r.kind == "gate"]) >= 5
    assert any(r.kind == "arena" for r in md.rooms)


def test_cli_and_png(tmp_path, capsys):
    out = tmp_path / "x.png"
    assert mapgen.main(["--gen", "cave", "--seed", "3", "--w", "40", "--h", "30", "--png", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 1000
    text = capsys.readouterr().out
    assert "connected" in text and "#" in text


def test_package_docstring():
    import emberdeep.scenarios as pkg
    assert pkg.__doc__ and "ascii_map" in pkg.__doc__
