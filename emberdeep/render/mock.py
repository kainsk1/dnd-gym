"""Synthetic scene + scripted fight used to develop, demo and test the world renderer without the engine.

    python -m emberdeep.render.mock --mode human
    python -m emberdeep.render.mock --mode rgb_array --out docs/screenshots
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

import numpy as np

from emberdeep.core.events import EventType as E
from emberdeep.core.types import DoorState, Material as M, Terrain as T
from emberdeep.render.api import (Decor, EntityView, FactionView, FrameState, LightSpec, StaticScene, ZoneView)

log = logging.getLogger(__name__)
W, H = 40, 28


def build_scene(seed: int = 3) -> StaticScene:
    rng = np.random.default_rng(seed)
    t = np.full((H, W), T.VOID, np.uint8)
    m = np.zeros((H, W), np.uint8)
    t[1:H - 1, 1:W - 1] = T.WALL
    m[1:H - 1, 1:W - 1] = M.STONE_WALL

    def rect(x0, y0, x1, y1, terr, mat):
        t[y0:y1 + 1, x0:x1 + 1] = terr
        m[y0:y1 + 1, x0:x1 + 1] = mat

    # great hall (flagstone, marble dais, carpet runner)
    rect(2, 2, 17, 13, T.FLOOR, M.STONE_FLOOR)
    rect(6, 4, 13, 11, T.FLOOR, M.MARBLE)
    rect(9, 2, 10, 13, T.FLOOR, M.CARPET_RED)
    for x, y in ((5, 3), (5, 12), (14, 3), (14, 12), (5, 7), (14, 8)):
        t[y, x] = T.PILLAR
        m[y, x] = M.STONE_WALL
    # study (wood) behind the south door
    rect(2, 15, 17, 25, T.FLOOR, M.WOOD_PLANKS)
    t[14, 9] = T.FLOOR; m[14, 9] = M.STONE_FLOOR           # doorway (south)
    rect(5, 20, 5, 20, T.LOW_WALL, M.BRICK_WALL)
    # corridor + door east
    rect(18, 8, 18, 8, T.FLOOR, M.STONE_FLOOR)            # doorway
    rect(19, 7, 21, 9, T.FLOOR, M.COBBLE)
    # cave (organic)
    yy, xx = np.mgrid[0:H, 0:W]
    blob = np.zeros((H, W), bool)
    for cx, cy, rx, ry in ((30, 8, 8.5, 6.5), (29, 17, 8.5, 6.0), (25, 23, 4.5, 3.2), (34, 13, 4.0, 8.0)):
        blob |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0 + rng.normal(0, 0.06, (H, W))
    blob[:, :22] = False
    blob[:2], blob[-2:] = False, False
    blob[:, -3:] = False
    t[blob] = T.FLOOR; m[blob] = M.CAVE_FLOOR
    wall_c = (t == T.WALL) & (xx >= 22)
    m[wall_c] = M.CAVE_WALL
    # cave materials
    dirt = blob & (rng.random((H, W)) < 0.0)
    for cx, cy, r in ((26, 10, 2.2), (33, 20, 2.0), (30, 12, 1.6)):
        mask = blob & (np.hypot(xx - cx, yy - cy) < r + rng.normal(0, .3, (H, W)))
        m[mask] = M.DIRT
    for cx, cy, r in ((36, 7, 1.8), (24, 6, 1.4)):
        mask = blob & (np.hypot(xx - cx, yy - cy) < r + rng.normal(0, .3, (H, W)))
        m[mask] = M.MOSS
    for cx, cy, r in ((33, 22, 2.0),):
        mask = blob & (np.hypot(xx - cx, yy - cy) < r + rng.normal(0, .3, (H, W)))
        m[mask] = M.GRASS
    # water pool
    pool = blob & (np.hypot((xx - 30.0) * 0.8, yy - 7.5) < 3.0)
    t[pool] = T.WATER_SHALLOW; m[pool] = M.WATER
    deep = blob & (np.hypot((xx - 30.0) * 0.8, yy - 7.5) < 1.5)
    t[deep] = T.WATER_DEEP; m[deep] = M.DEEP_WATER
    # lava fissure
    for x in range(25, 36):
        y0 = 15 + int((x - 25) * 0.25)
        for dy in range(0, 2 if x % 3 else 3):
            if blob[y0 + dy, x]:
                t[y0 + dy, x] = T.LAVA; m[y0 + dy, x] = M.LAVA
    # chasm
    ch = blob & (np.hypot(xx - 25, (yy - 24) * 1.1) < 1.9)
    t[ch] = T.PIT; m[ch] = M.CHASM
    scene_decor = [
        Decor("brazier", 3.5, 7.5), Decor("brazier", 16.5, 5.5), Decor("brazier", 16.5, 10.5), Decor("rug", 9.5, 7.5, 1.5708, 1.0, 1),
        Decor("statue", 3.5, 3.5, 0, 1, 0), Decor("statue", 16.5, 3.5, 0, 1, 1), Decor("banner", 7.5, 2.4, 0, 1, 0), Decor("banner", 12.5, 2.4, 0, 1, 0),
        Decor("barrel", 3.5, 11.5, 0, 1, 0), Decor("barrel", 4.5, 12.4, 0, 1, 1), Decor("crate", 3.5, 12.5, 0.3, 1, 1), Decor("bones", 15.5, 12.5, 0.3, 1, 1),
        Decor("blood_stain", 12.5, 9.5, 1, 1, 0), Decor("cobweb", 2.5, 2.5, 0, 1, 0), Decor("skull", 14.5, 2.7),
        Decor("table", 5.0, 17.0, 0, 1, 1), Decor("chair", 3.5, 17.5, 1.57), Decor("chair", 6.5, 17.5, -1.57), Decor("bookshelf", 5.0, 15.6, 0, 1, 0),
        Decor("bookshelf", 8.0, 15.6, 0, 1, 1), Decor("bed", 14.5, 17.5, 0, 1, 0), Decor("rug", 9.0, 21.0, 0, 1, 0), Decor("table", 11.0, 20.5, 0, 1, 2),
        Decor("candles", 11.0, 20.5), Decor("cauldron", 3.5, 23.5, 0, 1, 0), Decor("weapon_rack", 8.0, 24.5), Decor("anvil", 12.0, 24.5), Decor("cart", 14.0, 22.0),
        Decor("stairs_down", 16.5, 24.5), Decor("glyph", 6.5, 24.0, 0, 1, 1), Decor("coffin", 3.5, 20.5, 1.5708),
        Decor("torch_sconce", 7.5, 14.6), Decor("torch_sconce", 11.5, 14.6),
        Decor("crystal", 36.5, 12.5, 0, 1, 0), Decor("crystal", 24.5, 5.5, 0, 1, 2), Decor("rock", 26.5, 6.5, 0, 1, 0), Decor("rock", 35.5, 20.5, 0, 1, 1),
        Decor("mushrooms", 23.5, 12.5, 0, 1, 1), Decor("mushrooms", 36.5, 22.5, 0, 1, 0), Decor("campfire", 27.5, 12.5), Decor("tent", 25.0, 10.0, 0, 1, 0),
        Decor("bush", 33.5, 23.5, 0, 1, 1), Decor("bush", 32.5, 22.5, 0, 1, 0), Decor("puddle", 28.5, 11.5, 0, 1, 1), Decor("rubble", 22.5, 14.5, 0, 1, 1),
        Decor("fountain", 35.0, 5.0), Decor("tree", 36.0, 25.0, 0, 1, 1), Decor("altar", 23.5, 20.5, 0, 1, 0), Decor("moss_patch", 24.5, 8.5, 0, 1, 1),
        Decor("well", 21.0, 12.0), Decor("chains", 31.5, 10.5), Decor("cage", 37.0, 16.5), Decor("sarcophagus", 22.7, 18.0, 1.5708),
        Decor("throne", 16.0, 24.0, 0.0, 1, 0) if False else Decor("grate", 20.5, 8.5),
    ]
    lights = [
        LightSpec(3.5, 2.0, 5, 6, (1.0, 0.7, 0.4), 1.0, 0.18, 1.2, "torch"), LightSpec(9.5, 2.0, 5, 6, (1.0, 0.7, 0.4), 1.0, 0.18, 1.2, "torch"),
        LightSpec(15.5, 2.0, 5, 6, (1.0, 0.7, 0.4), 1.0, 0.18, 1.2, "torch"), LightSpec(2.0, 12.5, 4, 5, (1.0, 0.7, 0.4), 1.0, 0.18, 1.2, "torch"),
        LightSpec(3.5, 7.5, 5, 6, (1.0, 0.65, 0.3), 1.0, 0.25, 1.1, "brazier"), LightSpec(16.5, 5.5, 5, 6, (1.0, 0.65, 0.3), 1.0, 0.25, 1.1, "brazier"),
        LightSpec(16.5, 10.5, 5, 6, (1.0, 0.65, 0.3), 1.0, 0.25, 1.1, "brazier"), LightSpec(7.5, 14.6, 4, 6, (1.0, 0.72, 0.4), 0.9, 0.2, 1.2, "torch"),
        LightSpec(11.0, 20.5, 3, 5, (1.0, 0.8, 0.5), 0.9, 0.2, 1.0, "candle"), LightSpec(19.5, 7.0, 3, 4, (1.0, 0.7, 0.4), 0.9, 0.18, 1.2, "torch"),
        LightSpec(36.5, 12.5, 3, 5, (0.4, 0.7, 1.0), 1.0, 0.05, 1.0, "crystal"), LightSpec(24.5, 5.5, 3, 4, (0.8, 0.4, 1.0), 1.0, 0.05, 1.0, "crystal"),
        LightSpec(27.5, 12.5, 4, 6, (1.0, 0.6, 0.25), 1.0, 0.3, 1.0, "fire"), LightSpec(8.0, 24.5, 3, 4, (1.0, 0.7, 0.4), 0.8, 0.2, 1.2, "torch"),
        LightSpec(3.5, 23.5, 2.5, 3.5, (0.4, 1.0, 0.3), 0.7, 0.2, 1.0, "magic"),
    ]
    fac = [FactionView(0, "heroes", "The Ember Company", (74, 140, 230), True), FactionView(1, "goblins", "Gutter Goblins", (205, 62, 52), False)]
    return StaticScene("Mock: The Cinder Vault", W, H, t, m, None, scene_decor, lights, (0.075, 0.08, 0.12), "dungeon", fac,
                       ["fighter", "wizard", "cleric", "rogue"], "Synthetic showcase scene", seed)


class MockScript:
    """Builds a deterministic sequence of (FrameState, events) exercising most event types and VFX."""

    def __init__(self, scene: StaticScene | None = None):
        self.scene = scene or build_scene()
        self.ents: dict[int, EntityView] = {}
        self.doors: dict[tuple[int, int], str] = {(18, 8): "open", (9, 14): "closed"}
        self.zones: list[ZoneView] = []
        self.step_no = 0
        self.round = 1
        self.active = -1
        self.explored = {k: np.zeros((H, W), bool) for k in self.scene.agent_ids}
        self._init_entities()

    def _creature(self, eid, name, icon, x, y, fac, tint, agent=None, hp=24, ac=15, cls="", light=None, size=1):
        self.ents[eid] = EntityView(id=eid, kind="creature", name=name, x=x, y=y, size=size, icon=icon, faction=fac, tint=tint, agent_id=agent,
                                    controller="agent" if agent else "scripted", hp=hp, max_hp=hp, ac=ac, level=3 if agent else 0, class_name=cls or name,
                                    light=light, facing=0.0)

    def _obj(self, eid, kind, name, x, y, state="", icon=""):
        self.ents[eid] = EntityView(id=eid, kind=kind, name=name, x=x, y=y, icon=icon, state=state)

    def _init_entities(self):
        self._creature(1, "Brann", "barbarian", 12, 8, 0, (150, 90, 60), "fighter", 38, 18, "Fighter 3")
        self._creature(2, "Isolde", "wizard-face", 8, 8, 0, (90, 80, 160), "wizard", 18, 12, "Wizard 3", light=(2.0, 2.0))
        self._creature(3, "Mara", "holy-symbol", 7, 10, 0, (180, 160, 80), "cleric", 26, 16, "Cleric 3")
        self._creature(4, "Fen", "hooded-assassin", 10, 4, 0, (70, 100, 90), "rogue", 22, 14, "Rogue 3")
        self._creature(11, "Goblin A", "goblin-head", 16, 7, 1, (70, 120, 60), None, 12, 15, "Goblin")
        self._creature(12, "Goblin B", "goblin-head", 16, 9, 1, (80, 130, 55), None, 12, 15, "Goblin")
        self._creature(13, "Goblin C", "goblin-head", 15, 8, 1, (70, 110, 60), None, 12, 15, "Goblin")
        self._creature(14, "Skeleton", "raise-skeleton", 20, 8, 1, (140, 140, 120), None, 20, 13, "Skeleton")
        self._creature(15, "Ogre", "ogre", 26, 17, 1, (130, 100, 70), None, 59, 11, "Ogre", size=2)
        self._obj(20, "door", "Door", 18, 8, "open")
        self._obj(21, "door", "Door", 9, 14, "closed")
        self._obj(22, "chest", "Chest", 15, 24, "closed")
        self._obj(23, "loot", "Coins", 6, 12, "")
        self._obj(24, "lever", "Lever", 17, 12, "off")
        self.active = 4

    # -- helpers ------------------------------------------------------------------------------------
    def frame(self) -> FrameState:
        ents = [copy.copy(e) for e in self.ents.values()]
        for e in ents:
            e.conditions = list(e.conditions); e.effects = list(e.effects); e.statuses = list(e.statuses)
        vis_all = np.ones((H, W), bool)
        visible = {"omniscient": vis_all}
        explored = {"omniscient": vis_all}
        walls = np.isin(self.scene.terrain, (T.WALL, T.PILLAR, T.VOID))
        for (x, y), st in self.doors.items():
            if st in ("closed", "locked"):
                walls = walls.copy(); walls[y, x] = True
        for agent in self.scene.agent_ids:
            v = np.zeros((H, W), bool)
            e = next(c for c in self.ents.values() if c.agent_id == agent)
            for yy in range(max(0, e.y - 9), min(H, e.y + 10)):
                for xx in range(max(0, e.x - 11), min(W, e.x + 12)):
                    if (xx - e.x) ** 2 + (yy - e.y) ** 2 > 100:
                        continue
                    n = max(abs(xx - e.x), abs(yy - e.y), 1)
                    ok = True
                    for i in range(1, n):
                        px = int(round(e.x + (xx - e.x) * i / n)); py = int(round(e.y + (yy - e.y) * i / n))
                        if walls[py, px]:
                            ok = False
                            break
                    v[yy, xx] = ok
            self.explored[agent] |= v
            visible[agent] = v
            explored[agent] = self.explored[agent].copy()
        order = [4, 1, 11, 2, 12, 3, 13]
        return FrameState(self.step_no, self.round, "combat", self.active, order, ents, list(self.zones), visible, explored, None, dict(self.doors),
                          [], {}, {}, [[3, 0], [0, 3]], [("Defeat the goblins", "active")], None, False, "", {})

    def _ev(self, t, src=-1, dst=-1, amount=0, kind=-1, _d=None, **data):
        data = {**data, **(_d or {})}
        d = {"t": int(t), "type": E(t).name, "src": src, "dst": dst, "amount": amount, "kind": kind, "step": self.step_no, "round": self.round}
        if data:
            d["data"] = data
        return d

    def _move(self, eid, path, kind=0):
        evs = []
        e = self.ents[eid]
        for (x, y) in path:
            evs.append(self._ev(E.MOVE, eid, -1, 5, kind, frm=(e.x, e.y, 0), to=(x, y, 0)))
            e.x, e.y = x, y
        return evs

    def _hit(self, a, b, dmg, kind=0, crit=False, hit=True, name="Attack", ranged=False, weapon=None, vfx=None):
        evs = [self._ev(E.ATTACK, a, b, 0, 0, name=name, hit=hit, crit=crit, ranged=ranged, weapon=weapon, roll=15, total=18, ac=15, **({"vfx": vfx} if vfx else {}))]
        if hit:
            self.ents[b].hp = max(0, self.ents[b].hp - dmg)
            evs.append(self._ev(E.DAMAGE, a, b, dmg, kind, crit=crit, hp_after=self.ents[b].hp))
        else:
            evs.append(self._ev(E.MISS, a, b))
        return evs

    def _down(self, a, b, die=False):
        e = self.ents[b]
        e.hp = 0
        e.downed = True
        e.statuses = ["dying"]
        evs = [self._ev(E.DOWNED, a, b)]
        if die:
            e.dead, e.downed, e.alive = True, False, False
            e.statuses = ["dead"]
            evs.append(self._ev(E.DEATH, a, b, cause="damage"))
        return evs

    def _turn(self, eid):
        self.active = eid
        return [self._ev(E.TURN_START, eid, -1, self.round)]

    def steps(self):
        """Yield (label, frame, events) for each scripted step."""
        S = self
        out = []

        def push(label, evs):
            S.step_no += 1
            out.append((label, S.frame(), evs))

        out_events = [S._ev(E.EPISODE_START, scenario="mock")]
        push("start", out_events)
        order = [4, 1, 11, 2, 12, 3, 13]
        evs = [S._ev(E.COMBAT_START, order=order, rolls={}, surprised=[]), S._ev(E.ROUND_START, amount=1)] + S._turn(4)
        evs += S._hit(4, 11, 7, 1, False, True, "Shortbow", True, "shortbow", "arrow")
        evs.append(S._ev(E.TURN_END, 4))
        push("rogue_arrow", evs)
        # wizard magic missile + kill
        evs = S._turn(2)
        evs.append(S._ev(E.SPELL_CAST, 2, 11, 1, -1, spell="magic_missile", name="Magic Missile", targets=[11, 12], vfx="magic_missile"))
        S.ents[11].hp = 0; S.ents[12].hp -= 6
        evs += [S._ev(E.DAMAGE, 2, 11, 7, 12, hp_after=0), S._ev(E.DAMAGE, 2, 12, 6, 12, hp_after=S.ents[12].hp)]
        evs += S._down(2, 11, die=True)
        evs.append(S._ev(E.TURN_END, 2))
        push("magic_missile_kill", evs)
        # fighter charges, crit slash
        evs = S._turn(1)
        evs += S._move(1, [(13, 8), (14, 8)])
        evs += S._hit(1, 13, 9, 0, True, True, "Longsword", False, "longsword")
        S.ents[13].hp = 3
        evs.append(S._ev(E.TURN_END, 1))
        push("fighter_crit", evs)
        # cleric hold person + spirit guardians
        evs = S._turn(3)
        evs.append(S._ev(E.SPELL_CAST, 3, 12, 2, -1, spell="hold_person", name="Hold Person", targets=[12], vfx="hold_person"))
        evs.append(S._ev(E.SAVE, 3, 12, 4, 4, dc=13, roll=6, total=7, success=False))
        S.ents[12].conditions = ["paralyzed"]
        evs.append(S._ev(E.CONDITION_ADD, 3, 12, 0, 7, duration=10))
        push("hold_person", evs)
        zone_cells = [(x, y) for x in range(3, 12) for y in range(7, 14) if (x - 7) ** 2 + (y - 10) ** 2 <= 9.5 and S.scene.terrain[y, x] == T.FLOOR]
        zc = ZoneView(1, "spirit_guardians", zone_cells, (7.5, 10.5), 3.0, 3, False, 3)
        S.zones = [zc]
        evs = [S._ev(E.SPELL_CAST, 3, -1, 3, -1, spell="spirit_guardians", name="Spirit Guardians", targets=[], vfx="spirit_guardians",
                     area={"shape": 1, "size_ft": 15, "origin": (7, 10), "cells": zone_cells}),
               S._ev(E.ZONE_START, 3, -1, 0, -1, {"kind": "spirit_guardians"}, zone=1, cells=zone_cells, center=(7.5, 10.5), radius=3.0, harmful=False),
               S._ev(E.EFFECT_START, 3, 3, 0, -1, effect="bless", name="Bless", duration=10), S._ev(E.TURN_END, 3)]
        S.ents[3].effects = ["bless"]; S.ents[3].statuses = ["concentrating"]
        push("spirit_guardians", evs)
        # round 2: goblin moves & hits, one miss, rogue downed
        S.round = 2
        evs = [S._ev(E.ROUND_START, amount=2)] + S._turn(13)
        evs += S._move(13, [(14, 9), (13, 9)])
        evs += S._hit(13, 1, 5, 0, False, True, "Scimitar", False, "scimitar")
        evs += S._hit(13, 3, 0, 0, False, False, "Scimitar", False, "scimitar")
        evs.append(S._ev(E.TURN_END, 13))
        push("goblin_attacks", evs)
        evs = S._turn(14)
        evs += S._move(14, [(19, 8), (18, 8), (17, 8)])
        S.ents[4].hp = 0
        evs += S._hit(14, 4, 12, 0, False, True, "Shortsword", False, "shortsword")
        evs += S._down(14, 4)
        evs.append(S._ev(E.TURN_END, 14))
        push("skeleton_downs_rogue", evs)
        # wizard lightning bolt through the line
        evs = S._turn(2)
        S.ents[2].x, S.ents[2].y = 9, 8
        evs.append(S._ev(E.SPELL_CAST, 2, -1, 3, -1, spell="lightning_bolt", name="Lightning Bolt", targets=[14, 12, 13], vfx="lightning_bolt",
                         area={"shape": 4, "size_ft": 100, "origin": (9, 8), "direction": (1, 0), "cells": []}))
        for e, d in ((14, 14), (13, 3)):
            S.ents[e].hp = max(0, S.ents[e].hp - d)
            evs += [S._ev(E.SAVE, 2, e, 1, 1, dc=14, roll=5, total=7, success=False), S._ev(E.DAMAGE, 2, e, d, 6, hp_after=S.ents[e].hp)]
        evs += S._down(2, 13, die=True)
        evs.append(S._ev(E.TURN_END, 2))
        push("lightning_bolt", evs)
        # fireball on the doorway
        evs = S._turn(1)
        S.ents[2].x, S.ents[2].y = 8, 8
        evs.append(S._ev(E.SPELL_CAST, 2, -1, 3, -1, spell="fireball", name="Fireball", targets=[14, 12], vfx="fireball", point=(18, 8),
                         area={"shape": 1, "size_ft": 20, "origin": (18, 8), "cells": []}))
        S.ents[14].hp = 0
        evs += [S._ev(E.DAMAGE, 2, 14, 20, 5, hp_after=0), S._ev(E.DAMAGE, 2, 12, 12, 5, hp_after=0)]
        evs += S._down(2, 14, die=True) + S._down(2, 12, die=True)
        S.ents[12].conditions = []
        push("fireball", evs)
        # cleric healing word on downed rogue
        evs = S._turn(3)
        evs.append(S._ev(E.SPELL_CAST, 3, 4, 1, -1, spell="healing", name="Healing Word", targets=[4], vfx="healing"))
        S.ents[4].hp = 8; S.ents[4].downed = False; S.ents[4].statuses = []
        evs += [S._ev(E.HEAL, 3, 4, 8, hp_after=8), S._ev(E.REVIVED, 3, 4), S._ev(E.TURN_END, 3)]
        push("healing", evs)
        # web zone far east + misty step
        webc = [(x, y) for x in range(19, 22) for y in range(7, 10)]
        S.zones.append(ZoneView(2, "web", webc, (20.5, 8.5), 1.6, 2, True))
        evs = S._turn(2)
        evs.append(S._ev(E.SPELL_CAST, 2, -1, 2, -1, spell="web", name="Web", targets=[], vfx="web", point=(20, 8), area={"shape": 2, "size_ft": 15, "origin": (20, 8), "cells": webc}))
        evs.append(S._ev(E.ZONE_START, 2, -1, 0, -1, {"kind": "web"}, zone=2, cells=webc, center=(20.5, 8.5), radius=1.6, harmful=True))
        evs.append(S._ev(E.TELEPORT, 2, -1, 0, -1, frm=(8, 8, 0), to=(11, 6, 0), spell="misty_step"))
        S.ents[2].x, S.ents[2].y = 11, 6
        evs.append(S._ev(E.TURN_END, 2))
        push("web_misty_step", evs)
        # door closes, fighter bashes south door, chest, loot, gold
        evs = S._turn(4)
        S.doors[(18, 8)] = "closed"; S.ents[20].state = "closed"
        evs.append(S._ev(E.DOOR, 4, 20, 0, int(DoorState.CLOSED), forced=False))
        evs += S._move(4, [(9, 9), (9, 10), (9, 11), (9, 12)])
        S.zones = [z for z in S.zones if z.id != 2]
        evs.append(S._ev(E.ZONE_END, -1, -1, 0, -1, zone=2))
        push("close_door", evs)
        evs = S._turn(1)
        evs += S._move(1, [(10, 9), (10, 10), (9, 11), (9, 13)])
        S.doors[(9, 14)] = "open"; S.ents[21].state = "open"
        evs.append(S._ev(E.DOOR, 1, 21, 0, int(DoorState.OPEN), forced=True))
        evs.append(S._ev(E.SAY, 1, -1, 0, 3, ))
        push("fighter_opens_door", evs)
        evs = S._turn(4)
        evs += S._move(4, [(8, 12), (7, 12)])
        S.ents[4].gold = 0
        S.ents[1].gold = 50
        S.ents[23].state = "gone"; S.ents[23].x, S.ents[23].y = -5, -5
        evs.append(S._ev(E.PICKUP, 4, 23, 50, -1, items=[]))
        evs.append(S._ev(E.GIVE, 4, 3, 25, -1, items=[]))
        evs.append(S._ev(E.SAY, 3, -1, 0, 5))
        evs.append(S._ev(E.TURN_END, 4))
        push("loot_and_give", evs)
        evs = S._turn(1)
        evs += S._move(1, [(9, 15), (10, 16), (11, 17), (12, 18)] + [(13, 19 + i) for i in range(5)])
        S.ents[22].state = "open"
        evs.append(S._ev(E.CHEST_OPEN, 1, 22))
        push("chest_open", evs)
        return out


def run_script(r, script: MockScript, shots_dir: str | None = None, prefix: str = "world_mock", size_label: str = "", progress=print):
    """Drive a renderer through the script. In rgb_array mode screenshots are taken at key moments."""
    from PIL import Image
    r.set_scene(script.scene)
    shots = {  # step label -> list of simulated seconds at which to capture
        "start": [0.05], "rogue_arrow": [0.45], "magic_missile_kill": [0.55, 1.2], "fighter_crit": [1.0, 1.35], "hold_person": [0.95], "spirit_guardians": [1.2, 2.4],
        "goblin_attacks": [1.2], "lightning_bolt": [0.78, 0.9], "fireball": [0.9, 1.15, 1.6], "healing": [0.7], "web_misty_step": [1.3, 2.4], "chest_open": [2.2],
    }
    for label, frame, evs in script.steps():
        r.push(frame, evs)
        if r.mode == "human":
            r.play_until_idle()
            if not r.is_open():
                return
            continue
        caps = list(shots.get(label, []))
        t = 0.0
        n = 0
        while not r.director.idle and t < 60.0:
            r.step_sim(1 / 60.0)
            t += 1 / 60.0
            if caps and t >= caps[0] and shots_dir:
                caps.pop(0)
                n += 1
                img = r.capture()
                path = os.path.join(shots_dir, f"{prefix}_{label}_{n}.png")
                Image.fromarray(img).save(path)
                progress(f"saved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Emberdeep world renderer demo with a synthetic scene")
    ap.add_argument("--mode", default="human", choices=["human", "rgb_array"])
    ap.add_argument("--out", default="docs/screenshots")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--quality", default="medium")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--pov", default="omniscient")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    w, h = (int(v) for v in a.size.split("x"))
    from emberdeep.render.renderer import Renderer
    r = Renderer(a.mode, w, h, a.quality, hud=not a.no_hud, pov=a.pov)
    script = MockScript()
    if a.mode == "rgb_array":
        os.makedirs(a.out, exist_ok=True)
    try:
        run_script(r, script, a.out if a.mode == "rgb_array" else None)
        if a.mode == "human":
            while r.is_open():
                r.run_frame()
    finally:
        r.close()


if __name__ == "__main__":
    main()
