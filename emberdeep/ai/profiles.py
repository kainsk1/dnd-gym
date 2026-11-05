"""Behaviour profiles: plain parameter sets for the shared utility AI (``ai/utility.py`` + ``ai/tactics.py``).

A :class:`Profile` is data only.  The same scoring code serves every profile; profiles differ in weights, preferred
engagement range, risk aversion and a handful of boolean tactics.  See ``docs/AI.md`` ("How to add a profile").

Distances are in squares (5 ft).  Scores are in "expected hit points" (damage dealt / prevented / restored).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Profile:
    name: str
    style: str = "melee"                 # melee | ranged | caster | skirmish | support | ambush | leader | summon | idle
    # ---- target selection
    kill_w: float = 1.0                  # weight of the "kills the target" bonus
    focus_w: float = 0.6                 # prefer already wounded targets
    caster_w: float = 0.3                # prefer casters / healers / leaders
    guard_w: float = 0.0                 # prefer foes next to my squishy allies
    isolated_w: float = 0.0              # prefer foes with no friends nearby
    prone_w: float = 0.0                 # extra value for knocking prone / biting the prone
    finish_downed: float = 0.35          # relative value of attacking a downed (dying) foe
    # ---- movement
    band: tuple[float, float] | None = None   # preferred distance (squares) to the nearest foe, None = engage in melee
    safety: float = 0.0                  # weight of expected incoming damage at a cell
    cover_w: float = 0.0                 # weight of cover against foes
    ally_w: float = 0.0                  # penalty per square away from allies beyond ``ally_r``
    ally_r: float = 3.0
    flank_w: float = 0.0                 # bonus for cells adjacent to a foe that an ally also threatens
    move_margin: float = 0.6             # minimum gain before bothering to reposition
    hit_and_run: bool = False            # retreat after attacking when a cheap disengage exists
    kite: bool = False                   # step away from melee foes (accepting opportunity attacks only if worth it)
    hide_w: float = 0.0                  # value of hiding when a hide ability is available
    dash_ok: bool = True                 # dash to close the distance when nothing else is possible
    # ---- abilities
    slot_w: float = 1.0                  # cost weight of spending a spell slot (per slot level)
    heal_w: float = 1.0
    heal_thr: float = 0.55               # heal living allies below this hp fraction
    buff_w: float = 1.0
    control_w: float = 1.0
    aoe_w: float = 1.0
    friendly_fire_w: float = 1.6         # penalty multiplier for allies caught in an area
    conc_hold: float = 0.8               # fraction of the current concentration value that must be beaten
    stance_w: float = 1.0                # rage / surge / flurry style abilities
    # ---- morale
    retreat_hp: float = 0.0              # retreat towards allies when below this hp fraction (0 = never)
    flee_wound: bool = False             # flee at the first wound
    pack: bool = False                   # exploit pack tactics (prefer cells adjacent to a foe with an ally adjacent)
    patient: float = 0.0                 # ambush: rounds to wait before engaging when nothing is adjacent
    stay_back: bool = False              # leaders / healers keep behind the front line
    tags: tuple[str, ...] = field(default_factory=tuple)

    def with_(self, **kw) -> "Profile":
        return replace(self, **kw)


BRUTE = Profile("brute", style="melee", kill_w=1.0, focus_w=0.8, caster_w=0.2, safety=0.0, move_margin=0.5,
                finish_downed=0.5, retreat_hp=0.0)
SKIRMISHER = Profile("skirmisher", style="skirmish", band=(3, 9), safety=0.55, cover_w=0.5, ally_w=0.1, ally_r=5,
                     isolated_w=0.6, flank_w=0.6, hit_and_run=True, kite=True, hide_w=0.6, move_margin=0.6,
                     caster_w=0.3, focus_w=0.8, retreat_hp=0.3)
ARCHER = Profile("archer", style="ranged", band=(5, 14), safety=0.9, cover_w=1.0, ally_w=0.1, ally_r=6, kite=True,
                 move_margin=0.6, caster_w=0.4, focus_w=0.8, retreat_hp=0.3)
CASTER = Profile("caster", style="caster", band=(5, 16), safety=1.2, cover_w=0.9, ally_w=0.12, ally_r=5, kite=True,
                 slot_w=0.8, aoe_w=1.1, control_w=1.1, stay_back=True, move_margin=0.7, caster_w=0.5,
                 retreat_hp=0.4)
SUPPORT = Profile("support", style="support", band=(2, 10), safety=0.8, cover_w=0.6, ally_w=0.4, ally_r=3,
                  heal_w=1.8, heal_thr=0.65, buff_w=1.3, slot_w=0.6, stay_back=True, move_margin=0.7,
                  retreat_hp=0.35, kite=True)
BEAST = Profile("beast", style="melee", kill_w=1.0, focus_w=0.9, prone_w=1.0, flank_w=1.0, pack=True,
                safety=0.0, move_margin=0.4, isolated_w=0.5, finish_downed=0.7)
LEADER = Profile("leader", style="leader", band=(1, 6), safety=0.5, buff_w=1.4, ally_w=0.5, ally_r=3, cover_w=0.4,
                 stay_back=True, move_margin=0.7, heal_w=1.2, retreat_hp=0.25, guard_w=0.5)
COWARD = Profile("coward", style="melee", safety=1.5, flee_wound=True, band=(8, 20), kite=True, cover_w=0.5,
                 move_margin=0.5, retreat_hp=0.99)
AMBUSHER = Profile("ambusher", style="ambush", kill_w=1.1, focus_w=0.9, hide_w=2.0, patient=3, isolated_w=0.8,
                   safety=0.1, move_margin=0.5, prone_w=0.4)
BOSS = Profile("boss", style="melee", kill_w=1.2, focus_w=0.7, caster_w=0.5, aoe_w=1.3, control_w=1.2, safety=0.0,
               move_margin=0.4, finish_downed=0.2, guard_w=0.0)
SUMMON = Profile("summon", style="melee", kill_w=1.0, focus_w=0.8, safety=0.0, move_margin=0.4)
VIP = Profile("vip", style="vip", band=(8, 24), safety=2.0, cover_w=1.0, ally_w=0.5, ally_r=3, kite=True,
              move_margin=0.5, hide_w=1.0, retreat_hp=0.99)
IDLE = Profile("idle", style="idle")

PROFILES = {p.name: p for p in (BRUTE, SKIRMISHER, ARCHER, CASTER, SUPPORT, BEAST, LEADER, COWARD, AMBUSHER, BOSS,
                                SUMMON, VIP, IDLE)}
