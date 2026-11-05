"""Synthetic scene / frames / events for developing and testing the HUD without the engine.

A scripted mini fight: four heroes (fighter, cleric, rogue, wizard) against three goblins
and a bugbear.  ``MockFight.steps`` is a list of ``(FrameState, [event dict, ...])``.
Event dicts are produced with ``core.events.Event.to_dict()`` so they match the engine.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from emberdeep.core.events import Event, EventType as ET
from emberdeep.render.api import ActionDescriptor, EntityView, FactionView, FrameState, StaticScene, ZoneView
from emberdeep.render.hud_api import ViewInfo

W, H = 36, 22
PARTY, GOBLINS = 0, 1
HEROES = {1: "Aldric", 2: "Mara", 3: "Vex", 4: "Elowen"}


def make_scene() -> StaticScene:
    t = np.full((H, W), 2, np.uint8)
    t[2:H - 2, 2:W - 2] = 1                      # one big hall
    t[2:H - 2, 17] = 2                           # dividing wall ...
    t[10:13, 17] = 1                             # ... with a doorway
    t[4:6, 6:8] = 8                              # pillars
    t[15:17, 24:27] = 4                          # pool
    t[H - 3, W - 4] = 12                         # exit
    return StaticScene(
        name="The Goblin Warrens", width=W, height=H, terrain=t, material=np.zeros((H, W), np.uint8),
        factions=[FactionView(0, "party", "The Company", (98, 164, 240), True),
                  FactionView(1, "goblins", "Warren Goblins", (222, 84, 72), False)],
        agent_ids=["fighter", "cleric", "rogue", "wizard"],
        description="A torchlit warren beneath the hill.", theme="lair",
        comm_vocab=["Help!", "Hold the line!", "Truce?", "Fall back!", "Over here!", "Thank you"],
        regions={"exit": [(W - 4, H - 3)]},
    )


def _ev(v: EntityView, **kw) -> EntityView:
    for k, val in kw.items():
        setattr(v, k, val)
    return v


def make_entities() -> list[EntityView]:
    E = EntityView
    ents = [
        E(1, "creature", "Aldric", 8, 9, template="fighter", icon="visored-helm", faction=PARTY, agent_id="fighter",
          controller="agent", hp=31, max_hp=34, ac=18, level=3, class_name="Fighter 3", race_name="Human",
          speed=30, movement_left=30, gold=47, inventory=["Longsword", "Chain mail", "Shield", "Healing potion"],
          resources={"second_wind": (1, 1), "action_surge": (1, 1), "potions": (1, 2)},
          stats={"damage_dealt": 28, "damage_taken": 9, "kills": 1, "str": 16, "dex": 12, "con": 15, "int": 9,
                 "wis": 11, "cha": 10, "r_damage_dealt": 0.56, "r_kills": 0.5, "r_damage_taken": -0.09}),
        E(2, "creature", "Mara", 7, 12, template="cleric", icon="holy-symbol", faction=PARTY, agent_id="cleric",
          controller="agent", hp=22, max_hp=26, temp_hp=4, ac=16, level=3, class_name="Cleric 3", race_name="Dwarf",
          speed=25, movement_left=25, gold=12, concentration="bless", effects=["bless"],
          inventory=["Mace", "Holy symbol", "Scale mail"],
          resources={"slot_1": (3, 4), "slot_2": (1, 2), "channel_divinity": (1, 1), "potions": (0, 1)},
          stats={"healing_given": 14, "damage_dealt": 5, "str": 12, "dex": 10, "con": 14, "int": 10, "wis": 17,
                 "cha": 12, "r_healing_given": 0.28}),
        E(3, "creature", "Vex", 11, 7, template="rogue", icon="hooded-assassin", faction=PARTY, agent_id="rogue",
          controller="agent", hp=19, max_hp=21, ac=15, level=3, class_name="Rogue 3", race_name="Halfling",
          speed=25, movement_left=10, gold=133, effects=["dodge"], statuses=["hidden"],
          inventory=["Shortsword", "Shortbow", "Thieves' tools", "Dagger"],
          resources={"potions": (1, 1)},
          stats={"damage_dealt": 31, "gold_collected": 120, "kills": 2, "str": 8, "dex": 17, "con": 12, "int": 12,
                 "wis": 13, "cha": 11}),
        E(4, "creature", "Elowen", 6, 8, template="wizard", icon="wizard-face", faction=PARTY, agent_id="wizard",
          controller="agent", hp=11, max_hp=17, ac=12, level=3, class_name="Wizard 3", race_name="Elf",
          speed=30, movement_left=30, gold=64, conditions=["frightened"], concentration="hold_person",
          inventory=["Quarterstaff", "Spellbook", "Component pouch"],
          resources={"slot_1": (2, 4), "slot_2": (1, 2), "slot_3": (0, 2), "arcane_recovery": (1, 1)},
          stats={"damage_dealt": 22, "kills": 1, "str": 8, "dex": 14, "con": 13, "int": 17, "wis": 12, "cha": 10}),
        E(5, "creature", "Goblin", 14, 8, template="goblin", icon="goblin-head", faction=GOBLINS, hp=7, max_hp=7,
          ac=15, cr=0.25, class_name="Goblin", speed=30, tint=(120, 170, 90)),
        E(6, "creature", "Goblin Archer", 21, 6, template="goblin", icon="goblin-head", faction=GOBLINS, hp=4,
          max_hp=7, ac=15, cr=0.25, class_name="Goblin", speed=30, conditions=["prone"]),
        E(7, "creature", "Goblin Boss", 22, 12, template="goblin_boss", icon="goblin", faction=GOBLINS, hp=21,
          max_hp=21, ac=17, cr=1.0, class_name="Goblin Boss", speed=30),
        E(8, "creature", "Bugbear", 14, 11, template="bugbear", icon="ogre", faction=GOBLINS, hp=19, max_hp=27,
          ac=16, cr=1.0, class_name="Bugbear", speed=30, conditions=["poisoned"], effects=["hunters_mark"]),
        E(9, "door", "Door", 17, 11, template="door", icon="wooden-door", state="open"),
        E(10, "chest", "Chest", 29, 6, template="chest", icon="locked-chest", state="locked", value_gp=85),
    ]
    return ents


def make_visibility(ents: list[EntityView]) -> tuple[dict, dict]:
    full = np.ones((H, W), bool)
    vis = {"omniscient": full}
    exp = {"omniscient": full}
    v = np.zeros((H, W), bool)
    v[2:H - 2, 2:24] = True
    e = np.zeros((H, W), bool)
    e[2:H - 2, 2:W - 6] = True
    for k in ("fighter", "cleric", "rogue", "wizard", "faction:0"):
        vis[k] = v
        exp[k] = e
    return vis, exp


def make_frame(stage: str = "combat") -> FrameState:
    ents = make_entities()
    vis, exp = make_visibility(ents)
    combat = stage != "exploration"
    f = FrameState(
        step=7, round=3 if combat else 0, mode="combat" if combat else "exploration",
        active_id=3 if combat else -1,
        initiative=[3, 5, 4, 1, 8, 2, 7, 6] if combat else [1, 2, 3, 4],
        entities=ents, visible=vis, explored=exp,
        zones=[ZoneView(1, "stinking_cloud", [(15, 9), (15, 10), (16, 9), (16, 10)], (16.0, 10.0), 2.0, 4, True)],
        rewards={"fighter": 0.35, "cleric": -0.1, "rogue": 0.8, "wizard": 0.0},
        returns={"fighter": 2.4, "cleric": 1.1, "rogue": 3.2, "wizard": -0.6},
        relations=[[3, 0], [0, 3]],
        objectives=[("Defeat the Warren Goblins", "active"), ("Open the chest", "active"),
                    ("Find the hidden door", "done")],
        annotations={"rogue": {"value": 1.82, "top_actions": [("Shortsword attack Goblin", 0.46),
                                                              ("Move NE", 0.21), ("Hide", 0.15),
                                                              ("Dash", 0.08), ("End turn", 0.05)],
                               "note": "Sneak attack available (ally adjacent)"},
                     "fighter": {"value": 0.9, "top_actions": [("Longsword attack Bugbear", 0.7),
                                                                ("Second Wind", 0.2)], "note": ""}},
        metrics={"gini_gold": 0.31, "betrayals": 1, "heals_given": 14, "truces_formed": 1, "team_damage": 3.0,
                 "cooperation_index": 0.72, "deaths": 1},
    )
    if not combat:
        for e in f.entities:
            if e.faction == GOBLINS:
                e.hidden = True
    return f


def make_view(size: tuple[int, int], frame: FrameState, scene: StaticScene, pov: str = "omniscient",
              hovered: int = -1, selected: int = -1, human: bool = False, options: dict | None = None) -> ViewInfo:
    w, h = size
    cp = h / 20.0
    ox = w / 2 - 14.0 * cp
    oy = h / 2 - 9.6 * cp

    def w2s(x: float, y: float) -> tuple[float, float]:
        return ox + x * cp, oy + y * cp

    def s2w(px: float, py: float) -> tuple[float, float]:
        return (px - ox) / cp, (py - oy) / cp

    es = {}
    for e in frame.entities:
        cx, cy = w2s(e.x + e.size / 2, e.y + e.size / 2)
        es[e.id] = (cx, cy, cp * 0.42 * e.size)
    return ViewInfo(
        screen_size=size, world_to_screen=w2s, screen_to_world=s2w, cell_px=cp, pov=pov,
        pov_options=["omniscient"] + list(scene.agent_ids) + [f"faction:{f.index}" for f in scene.factions],
        entity_screen=es, visible_entities={e.id for e in frame.entities}, hovered_eid=hovered,
        selected_eid=selected, human_turn=human, options=options if options is not None else {})


# --------------------------------------------------------------------------- events
def ev(t: ET, src: int = -1, dst: int = -1, amount: int = 0, kind: int = -1, step: int = 7, rnd: int = 3,
       **data: Any) -> dict:
    return Event(int(t), src, dst, amount, kind, data or None, step, rnd).to_dict()


def fight_events() -> list[list[dict]]:
    """Event batches, one per step."""
    S = ev
    return [
        [S(ET.EPISODE_START, scenario="goblin_warren", seed=1)],
        [S(ET.SPOTTED, 3, 5), S(ET.COMBAT_START, order=[3, 5, 4, 1, 8, 2, 7, 6],
                                rolls={"3": 19, "5": 16, "4": 14, "1": 12, "8": 11, "2": 9, "7": 7, "6": 5},
                                surprised=[6])],
        [S(ET.ROUND_START, amount=1, rnd=1), S(ET.TURN_START, 3, amount=1, rnd=1),
         S(ET.MOVE, 3, amount=15, kind=0, frm=[9, 7, 0], to=[11, 7, 0], rnd=1),
         S(ET.ATTACK, 3, 5, kind=0, ability="shortsword", name="Shortsword", roll=17, rolls=[17, 6], bonus=5,
           total=22, ac=15, hit=True, crit=False, fumble=False, adv=1, cover=0, ranged=False, weapon="shortsword",
           opportunity=False, rnd=1),
         S(ET.DAMAGE, 3, 5, amount=6, kind=7, raw=6, mod=None, crit=False, source="shortsword", hp_after=1, rnd=1),
         S(ET.DAMAGE, 3, 5, amount=3, kind=7, raw=3, mod=None, crit=False, source="sneak_attack", hp_after=0, rnd=1)],
        [S(ET.DEATH, 3, 5, cause="damage", rnd=1), S(ET.TURN_END, 3, rnd=1), S(ET.TURN_START, 5, amount=1, rnd=1),
         S(ET.ATTACK, 5, 1, kind=0, ability="scimitar", name="Scimitar", roll=4, rolls=[4], bonus=4, total=8, ac=18,
           hit=False, crit=False, fumble=False, adv=0, cover=0, ranged=False, weapon="scimitar", rnd=1),
         S(ET.MISS, 5, 1, rnd=1)],
        [S(ET.TURN_START, 4, amount=1, rnd=1),
         S(ET.SPELL_CAST, 4, -1, amount=3, spell="fireball", name="Fireball", targets=[8, 7],
           point=[16, 10], area={"shape": "sphere", "size_ft": 20, "origin": [16, 10], "cells": [[16, 10]]},
           vfx="fireball", rnd=1),
         S(ET.SAVE, 4, 8, kind=1, dc=14, roll=7, total=9, success=False, source="fireball", rnd=1),
         S(ET.DAMAGE, 4, 8, amount=22, kind=3, raw=22, mod=None, crit=False, source="fireball", hp_after=19, rnd=1),
         S(ET.SAVE, 4, 7, kind=1, dc=14, roll=15, total=17, success=True, source="fireball", rnd=1),
         S(ET.DAMAGE, 4, 7, amount=4, kind=3, raw=8, mod="resisted", crit=False, source="fireball", hp_after=17,
           rnd=1),
         S(ET.CONDITION_ADD, 4, 8, kind=9, duration=10, rnd=1)],
        [S(ET.TURN_START, 2, amount=2, rnd=2),
         S(ET.SPELL_CAST, 2, 1, amount=1, spell="cure_wounds", name="Cure Wounds", targets=[1], vfx="healing",
           rnd=2),
         S(ET.HEAL, 2, 1, amount=9, source="cure_wounds", hp_after=31, rnd=2),
         S(ET.EFFECT_START, 2, 1, effect="bless", name="Bless", duration=10, rnd=2),
         S(ET.SAY, 2, kind=5, rnd=2)],
        [S(ET.ATTACK, 8, 4, kind=0, ability="morningstar", name="Morningstar", roll=20, rolls=[20, 11], bonus=4,
           total=24, ac=12, hit=True, crit=True, fumble=False, adv=-1, ranged=False, rnd=2),
         S(ET.DAMAGE, 8, 4, amount=14, kind=1, raw=14, mod=None, crit=True, source="morningstar", hp_after=0,
           temp_absorbed=0, rnd=2),
         S(ET.DOWNED, 8, 4, rnd=2), S(ET.DEATH_SAVE, -1, 4, roll=12, success=True, successes=1, failures=0, rnd=2),
         S(ET.CONCENTRATION_CHECK, -1, 4, dc=10, roll=6, total=8, success=False, spell="hold_person", rnd=2),
         S(ET.CONCENTRATION_END, -1, 4, spell="hold_person", reason="damage", rnd=2)],
        [S(ET.TRUCE_OFFER, 7, 1, factions=[1, 0], rnd=2), S(ET.BETRAYAL, 1, 7, relation=2, rnd=2),
         S(ET.PICKUP, 3, 10, amount=85, items=["gem"], rnd=3), S(ET.GOLD_CHANGE, -1, 3, amount=85, reason="loot", rnd=3),
         S(ET.OBJECTIVE, 3, kind=1, name="Find the hidden door", success=True, rnd=3)],
    ]


def _fight_frames() -> list[FrameState]:
    """Frames after each batch (HP progressions)."""
    frames = []
    base = make_frame("combat")
    for i in range(9):
        f = copy.deepcopy(base)
        f.step = i
        k = (i + 1) / 9.0
        f.returns = {a: round(v * k + 0.25 * np.sin(i * 1.7 + len(a)), 3) for a, v in f.returns.items()}
        if i == 8:
            f.returns = dict(base.returns)
        hp = {5: 7, 6: 4, 7: 21, 8: 27, 4: 17, 1: 22, 2: 26}
        if i >= 2:
            hp[5] = 0
        if i >= 4:
            hp[8], hp[7] = 19, 17
        if i >= 5:
            hp[1] = 31
        if i >= 6:
            hp[4] = 0
        for e in f.entities:
            if e.id in hp:
                e.hp = hp[e.id]
            if e.id == 5 and i >= 2:
                e.dead, e.alive = True, False
            if e.id == 4 and i >= 6:
                e.downed, e.death_saves = True, (1, 0)
        if i == 0:
            f.mode, f.round = "exploration", 0
        frames.append(f)
    return frames


class MockFight:
    def __init__(self) -> None:
        self.scene = make_scene()
        self.steps = list(zip(_fight_frames(), fight_events()))


# --------------------------------------------------------------------------- every event type
def all_event_dicts() -> list[dict]:
    """One representative event dict per EventType (some with sparse data to test tolerance)."""
    out = []
    atk = dict(ability="longsword", name="Longsword", roll=15, rolls=[15, 9], bonus=5, total=20, ac=15, hit=True,
               crit=False, fumble=False, adv=1, cover=0, ranged=False, weapon="longsword", opportunity=False)
    gen: dict[ET, dict] = {
        ET.EPISODE_START: dict(src=-1, dst=-1, data=dict(scenario="x", seed=1)),
        ET.COMBAT_START: dict(data=dict(order=[1, 5, 3], rolls={"1": 10, "5": 9, "3": 4}, surprised=[5])),
        ET.COMBAT_END: dict(data=dict(reason="all hostile defeated")),
        ET.ROUND_START: dict(amount=2),
        ET.TURN_START: dict(src=1, amount=2), ET.TURN_END: dict(src=1),
        ET.EPISODE_END: dict(data=dict(reason="objective", winners=[0])),
        ET.MOVE: dict(src=1, amount=10, kind=0, data=dict(frm=[8, 9, 0], to=[9, 9, 0])),
        ET.FORCED_MOVE: dict(src=1, dst=5, amount=2, data=dict(frm=[14, 8, 0], to=[16, 8, 0])),
        ET.TELEPORT: dict(src=4, data=dict(frm=[6, 8, 0], to=[9, 8, 0], spell="misty_step")),
        ET.FALL: dict(dst=5, amount=7, data=dict(height_ft=20)),
        ET.STAND_UP: dict(src=6), ET.ESCAPE: dict(src=3),
        ET.ATTACK: dict(src=1, dst=5, kind=0, data=atk),
        ET.DAMAGE: dict(src=1, dst=5, amount=9, kind=3, data=dict(raw=9, mod="resisted", crit=False, source="x",
                                                                 hp_after=1, temp_absorbed=2)),
        ET.HEAL: dict(src=2, dst=1, amount=8, data=dict(source="cure_wounds", hp_after=30)),
        ET.TEMP_HP: dict(src=2, dst=1, amount=5),
        ET.SAVE: dict(src=4, dst=5, kind=1, data=dict(dc=14, roll=8, total=10, success=False, source="fireball")),
        ET.CHECK: dict(dst=3, kind=100 + 1, data=dict(dc=15, roll=12, total=19, success=True, purpose="lockpick")),
        ET.MISS: dict(src=5, dst=1),
        ET.SPELL_CAST: dict(src=4, dst=5, amount=1, data=dict(spell="magic_missile", name="Magic Missile",
                                                             targets=[5], point=None, area=None, vfx="magic_missile")),
        ET.ABILITY_USE: dict(src=1, dst=-1, data=dict(ability="second_wind", name="Second Wind", vfx="second_wind")),
        ET.EFFECT_START: dict(src=2, dst=1, data=dict(effect="bless", name="Bless", duration=10)),
        ET.EFFECT_END: dict(dst=1, data=dict(effect="bless")),
        ET.CONCENTRATION_CHECK: dict(dst=2, data=dict(dc=10, roll=14, total=17, success=True, spell="bless")),
        ET.CONCENTRATION_END: dict(dst=2, data=dict(spell="bless", reason="incapacitated")),
        ET.ZONE_START: dict(src=4, data=dict(zone=1, kind="fog_cloud", cells=[[1, 1]], center=[1, 1], radius=2.0,
                                             harmful=False)),
        ET.ZONE_END: dict(data=dict(zone=1)), ET.ZONE_MOVE: dict(data=dict(zone=1, cells=[[2, 2]], center=[2, 2])),
        ET.SUMMON: dict(src=4, dst=11), ET.TRANSFORM: dict(dst=2, data=dict(into="bear")),
        ET.REACTION: dict(src=1, dst=5, kind=1, data=dict(ability="opportunity_attack", name="Opportunity Attack")),
        ET.COUNTERSPELL: dict(src=4, dst=7, data=dict(spell="fireball", success=True)),
        ET.RECHARGE: dict(src=8, data=dict(ability="breath", roll=6, success=True)),
        ET.CONDITION_ADD: dict(src=4, dst=5, kind=9, data=dict(duration=10)),
        ET.CONDITION_REMOVE: dict(dst=5, kind=9),
        ET.DOWNED: dict(src=8, dst=4), ET.DEATH: dict(src=3, dst=5, data=dict(cause="damage")),
        ET.DEATH_SAVE: dict(dst=4, data=dict(roll=3, success=False, successes=1, failures=1)),
        ET.STABILIZED: dict(src=2, dst=4), ET.REVIVED: dict(src=2, dst=4), ET.EXHAUSTION: dict(dst=1, amount=2),
        ET.SURRENDER: dict(src=6), ET.FLEE: dict(src=6),
        ET.PICKUP: dict(src=3, dst=10, amount=40, data=dict(items=["gem"])),
        ET.GIVE: dict(src=3, dst=1, amount=20, data=dict(items=[])),
        ET.USE_ITEM: dict(src=1, dst=1, data=dict(item="healing_potion")),
        ET.DROP: dict(src=6, data=dict(items=["dagger"], gold=3, at=[20, 6])),
        ET.EQUIP: dict(src=1, data=dict(item="shield", slot="off_hand")),
        ET.DOOR: dict(src=1, dst=9, kind=0, data=dict(forced=True)),
        ET.CHEST_OPEN: dict(src=3, dst=10),
        ET.LOCKPICK: dict(src=3, dst=10, data=dict(success=True, dc=15, roll=14)),
        ET.TRAP_TRIGGER: dict(src=12, dst=3, data=dict(kind="dart")),
        ET.TRAP_DETECT: dict(src=3, dst=12), ET.TRAP_DISARM: dict(src=3, dst=12, data=dict(success=True)),
        ET.LEVER: dict(src=1, dst=13, data=dict(state="on")),
        ET.INTERACT: dict(src=2, dst=14, data=dict(kind="altar", result="blessed")),
        ET.GOLD_CHANGE: dict(dst=3, amount=-15, data=dict(reason="trade")),
        ET.HIDE: dict(src=3, data=dict(roll=14, total=20, success=True)),
        ET.REVEALED: dict(dst=3, data=dict(reason="attacked")),
        ET.SPOTTED: dict(src=5, dst=3), ET.SEARCH: dict(src=3, data=dict(roll=10, total=15, found=[13])),
        ET.SAY: dict(src=2, kind=1), ET.TRUCE_OFFER: dict(src=7, dst=1, data=dict(factions=[1, 0])),
        ET.TRUCE_ACCEPT: dict(src=1, dst=7, data=dict(factions=[0, 1])),
        ET.RELATION_CHANGE: dict(data=dict(a=0, b=1, old=0, new=2)),
        ET.BETRAYAL: dict(src=1, dst=7, data=dict(relation=2)),
        ET.PARLEY: dict(src=1, dst=7, data=dict(success=False, dc=15, roll=6)),
        ET.REST_VOTE: dict(src=1, amount=1), ET.REST: dict(data=dict(kind="short", participants=[1, 2, 3])),
        ET.OBJECTIVE: dict(src=3, kind=0, data=dict(name="Open the chest", team=0, success=True)),
        ET.LOG: dict(data=dict(text="A cold wind rolls out of the dark.", level="info")),
    }
    for t in ET:
        kw = gen.get(t, {})
        out.append(Event(int(t), kw.get("src", -1), kw.get("dst", -1), kw.get("amount", 0), kw.get("kind", -1),
                         kw.get("data"), 7, 2).to_dict())
    return out


# --------------------------------------------------------------------------- human choice
def make_actions() -> list[ActionDescriptor]:
    A = ActionDescriptor
    acts = [
        A(0, "end_turn", "End Turn"),
        A(1, "ability", "Shortsword", "shortsword", "broadsword", "action", target_eid=5, hit_chance=0.72,
          expected_damage=8.5, tooltip="Melee weapon attack. Finesse; sneak attack applies with advantage."),
        A(2, "ability", "Shortsword", "shortsword", "broadsword", "action", target_eid=8, hit_chance=0.55,
          expected_damage=8.5),
        A(3, "ability", "Shortbow", "shortbow", "bow-arrow", "action", target_eid=5, hit_chance=0.8,
          expected_damage=7.0, tooltip="Ranged weapon attack, 80/320 ft."),
        A(4, "ability", "Shortbow", "shortbow", "bow-arrow", "action", target_eid=8, hit_chance=0.6,
          expected_damage=7.0),
        A(5, "ability", "Dash", "dash", "sprint", "action"),
        A(6, "ability", "Hide", "hide", "cowled", "bonus", tooltip="Cunning Action: try to vanish from sight."),
        A(7, "ability", "Disengage", "disengage", "run", "bonus"),
        A(8, "ability", "Healing Potion", "healing_potion", "health-potion", "bonus", target_eid=1),
        A(9, "ability", "Fireball", "fireball", "fire-spell-cast", "action", target_cell=(15, 10),
          area_cells=[(14, 9), (15, 9), (16, 9), (14, 10), (15, 10), (16, 10), (14, 11), (15, 11), (16, 11)],
          expected_damage=28.0, tooltip="20-ft burst, DEX save for half."),
        A(10, "ability", "Fireball", "fireball", "fire-spell-cast", "action", target_cell=(20, 12),
          area_cells=[(19, 11), (20, 11), (21, 11), (19, 12), (20, 12), (21, 12), (19, 13), (20, 13), (21, 13)],
          expected_damage=28.0),
        A(11, "move", "Move N", direction=0, target_cell=(11, 6), cost="movement"),
        A(12, "move", "Move E", direction=2, target_cell=(12, 7), cost="movement"),
        A(13, "move", "Move S", direction=4, target_cell=(11, 8), cost="movement"),
        A(14, "move", "Move W", direction=6, target_cell=(10, 7), cost="movement"),
        A(15, "approach", "Approach Goblin", "shortsword", "", "movement", target_eid=5, target_cell=(13, 8)),
        A(16, "say", "Say: Help!", cost="free"),
        A(17, "say", "Say: Hold the line!", cost="free"),
    ]
    dis = ActionDescriptor(18, "ability", "Second Wind", "second_wind", "heart-plus", "bonus")
    dis.enabled = False  # type: ignore[attr-defined]
    acts.append(dis)
    return acts
