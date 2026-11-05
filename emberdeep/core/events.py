"""Engine event stream (contract).

Every rule resolution emits ``Event`` records into ``World.events``. Events are
consumed by:

* the reward system (``emberdeep.env.rewards``) - uses only the compact fields
  ``t, src, dst, amount, kind``;
* the metrics system (social outcome metrics);
* the renderer / combat log / replays - additionally use ``data``.

Events are cheap (``__slots__``) and JSON-serialisable via ``to_dict``.

Entity references (``src`` / ``dst``) are *entity ids* (``eid``): creatures and
world objects share one id space (see DESIGN.md §4.1). ``-1`` means none.

Field conventions per type are documented next to each ``EventType`` member as
``src -> dst | amount | kind | data keys``. Renderer and log code must tolerate
missing ``data`` keys.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any


class EventType(IntEnum):
    # --- flow -----------------------------------------------------------------
    EPISODE_START = 0     # - | - | - | data: scenario, seed
    COMBAT_START = 1      # - | - | - | data: order [eid...], rolls {eid: init}, surprised [eid...]
    COMBAT_END = 2        # - | - | - | data: reason
    ROUND_START = 3       # - | amount=round number
    TURN_START = 4        # src=creature | amount=round
    TURN_END = 5          # src=creature
    EPISODE_END = 6       # - | - | - | data: reason, winners [faction ids]

    # --- movement ---------------------------------------------------------------
    MOVE = 10             # src | amount=feet spent | kind=0 walk,1 fly,2 swim,3 climb,4 crawl | data: frm (x,y,z), to (x,y,z)
    FORCED_MOVE = 11      # src=pusher -> dst=moved | amount=squares | data: frm, to
    TELEPORT = 12         # src | data: frm, to, spell
    FALL = 13             # dst | amount=damage | data: height_ft
    STAND_UP = 14         # src
    ESCAPE = 15           # src left the map via an exit

    # --- attacks & damage -------------------------------------------------------
    ATTACK = 20           # src -> dst | kind=AttackKind | data: ability (id), name, roll (d20 kept), rolls [d20s], bonus, total,
                          #   ac, hit (bool), crit (bool), fumble (bool), adv (-1,0,1), cover (Cover), ranged (bool), weapon (id or None), opportunity (bool)
    DAMAGE = 21           # src -> dst | amount=final damage | kind=DamageType | data: raw, mod ('resisted','vulnerable','immune',None),
                          #   crit (bool), source (ability/spell id), hp_after, temp_absorbed
    HEAL = 22             # src -> dst | amount | data: source, hp_after
    TEMP_HP = 23          # src -> dst | amount
    SAVE = 24             # src=forcer -> dst=saver | kind=Ability | data: dc, roll, total, success (bool), source
    CHECK = 25            # dst=checker | kind=Skill or (100+Ability) | data: dc, roll, total, success, contest_with (eid), purpose
    MISS = 26             # src -> dst (convenience duplicate of ATTACK hit=False for log/VFX)

    # --- spells & abilities -----------------------------------------------------
    SPELL_CAST = 30       # src -> dst (primary target or -1) | amount=slot level (0 cantrip) | data: spell (id), name, targets [eid],
                          #   point (x,y) or None, area {shape, size_ft, origin (x,y), direction (dx,dy), cells [(x,y)...]}, vfx (key)
    ABILITY_USE = 31      # src -> dst | data: ability (id), name, vfx — class features, basic actions (Dash, Dodge...), monster specials
    EFFECT_START = 32     # src=source -> dst=bearer | data: effect (id), name, duration (rounds or None)
    EFFECT_END = 33       # dst=bearer | data: effect (id)
    CONCENTRATION_CHECK = 34  # dst | data: dc, roll, total, success, spell
    CONCENTRATION_END = 35    # dst | data: spell, reason
    ZONE_START = 36       # src=creator | data: zone id (int), kind (str), cells [(x,y)], center, radius, harmful (bool)
    ZONE_END = 37         # data: zone id
    ZONE_MOVE = 38        # data: zone id, cells, center
    SUMMON = 39           # src=summoner -> dst=new creature
    TRANSFORM = 40        # dst | data: into (template id) / revert (bool)
    REACTION = 41         # src -> dst | kind=DecisionKind | data: ability (id), name
    COUNTERSPELL = 42     # src -> dst | data: spell countered, success
    RECHARGE = 43         # src | data: ability, roll, success

    # --- conditions & life -------------------------------------------------------
    CONDITION_ADD = 50    # src=source -> dst | kind=Condition | data: duration, save
    CONDITION_REMOVE = 51 # dst | kind=Condition
    DOWNED = 52           # src=attacker -> dst | (dropped to 0 HP, unconscious & dying)
    DEATH = 53            # src=killer -> dst | data: cause ('damage','death_saves','massive','instakill')
    DEATH_SAVE = 54       # dst | data: roll, success (bool), successes, failures
    STABILIZED = 55       # src=helper (or -1) -> dst
    REVIVED = 56          # src -> dst (regained HP from 0)
    EXHAUSTION = 57       # dst | amount=new level
    SURRENDER = 58        # src surrenders (morale)
    FLEE = 59             # src starts fleeing (morale)

    # --- items, objects, economy ---------------------------------------------------
    PICKUP = 60           # src | dst=loot/chest entity | amount=gold gp | data: items [item ids]
    GIVE = 61             # src -> dst | amount=gold | data: items [item ids]
    USE_ITEM = 62         # src -> dst | data: item (id)
    DROP = 63             # src | data: items, gold, at (x,y)
    EQUIP = 64            # src | data: item, slot
    DOOR = 65             # src -> dst=door | kind=DoorState (new) | data: forced (bool)
    CHEST_OPEN = 66       # src -> dst=chest
    LOCKPICK = 67         # src -> dst | data: success, dc, roll
    TRAP_TRIGGER = 68     # src=trap -> dst=victim | data: trap kind
    TRAP_DETECT = 69      # src=detector -> dst=trap
    TRAP_DISARM = 70      # src -> dst | data: success
    LEVER = 71            # src -> dst=lever | data: state
    INTERACT = 72         # src -> dst=object | data: kind (altar/fountain/relic...) , result
    GOLD_CHANGE = 73      # dst | amount (signed) | data: reason
    MOUNT = 74            # src=rider -> dst=mount | data: free (bool)
    DISMOUNT = 75         # src=rider -> dst=mount | data: reason ('voluntary','fell','mount_down','forced'), prone (bool)

    # --- perception & stealth --------------------------------------------------------
    HIDE = 80             # src | data: roll, total, success
    REVEALED = 81         # dst (stopped being hidden) | data: reason
    SPOTTED = 82          # src (observer) -> dst (newly noticed creature)
    SEARCH = 83           # src | data: roll, total, found [eid]

    # --- social / diplomacy ------------------------------------------------------------
    SAY = 90              # src | kind=token id
    TRUCE_OFFER = 91      # src -> dst | data: factions (a, b)
    TRUCE_ACCEPT = 92     # src -> dst | data: factions (a, b)
    RELATION_CHANGE = 93  # data: a, b (faction indices), old, new (Relation)
    BETRAYAL = 94         # src attacked dst despite ALLIED/TRUCE relation | data: relation
    PARLEY = 95           # src -> dst | data: success, dc, roll
    REST_VOTE = 96        # src | amount=1 vote / 0 withdraw
    REST = 97             # data: kind ('short'), participants [eid]

    # --- scenario ------------------------------------------------------------------------
    OBJECTIVE = 100       # src=completer | kind=objective index | data: name, team/faction, success
    LOG = 101             # free-text narration for the combat log | data: text, level


class Event:
    __slots__ = ("t", "src", "dst", "amount", "kind", "data", "step", "round")

    def __init__(self, t: int, src: int = -1, dst: int = -1, amount: int = 0, kind: int = -1,
                 data: dict[str, Any] | None = None, step: int = 0, round: int = 0):
        self.t = t
        self.src = src
        self.dst = dst
        self.amount = amount
        self.kind = kind
        self.data = data
        self.step = step
        self.round = round

    def to_dict(self) -> dict[str, Any]:
        d = {"t": int(self.t), "type": EventType(self.t).name, "src": self.src, "dst": self.dst,
             "amount": self.amount, "kind": self.kind, "step": self.step, "round": self.round}
        if self.data:
            d["data"] = _jsonable(self.data)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        return Event(d["t"], d.get("src", -1), d.get("dst", -1), d.get("amount", 0), d.get("kind", -1),
                     d.get("data"), d.get("step", 0), d.get("round", 0))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"Event({EventType(self.t).name}, src={self.src}, dst={self.dst}, amount={self.amount}, "
                f"kind={self.kind}, data={self.data})")


def _jsonable(x: Any) -> Any:
    """Convert numpy scalars / tuples / enums recursively into JSON-friendly types."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return x.item()
        except Exception:  # noqa: BLE001
            return str(x)
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)
