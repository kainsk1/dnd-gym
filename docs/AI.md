# Scripted AI (`emberdeep/ai`)

## API
```python
from emberdeep.ai import get_policy, policy_for, scripted_decider
world.auto_decider = scripted_decider                 # reactions / riders / legendary actions
act = policy_for(world, eid).act(world, eid)          # ONE atomic, legal act for the current decider
world.step(act)
```
`Policy.act / decide / reset`; `get_policy(name, seed=None)` names: `random brute skirmisher archer caster support beast
leader coward ambusher boss summon vip idle hero hero_<class>`. `policy_for` precedence: `vip` tag > `creature.ai`
(SpawnSpec.ai / MonsterDef.ai_profile) > PC class (`hero_<class>`) > summoner (`summon`) > `boss` tag > `brute`.
Scripted policies are stateless and shared; `random` owns a seeded generator (or derives one from world seed / step).

## Architecture
| module | role |
|---|---|
| `policy.py` | `UtilityPolicy`: gate (asleep / guard:<region> / stationary), social layer, survey, movement, pick best |
| `abinfo.py` | classifies any runtime `Ability` (attack, aoe, control, debuff, heal, buff, mark, summon, escape, teleport, stance, ...) from ids / tags / SpellDef / AttackDef; no hard-coded indices except the 26 universal ids |
| `scene.py` | per-act situation (`foes`, `standing`, `vis`, `allies`, downed ...), estimate memoisation, foe offence model |
| `utility.py` | candidate enumeration + scoring (`Survey`, `Cand`) |
| `tactics.py` | vectorised cell scoring: approach, kite, cover, flank, retreat, hide, disengage, teleport, flee |
| `social.py` | morale, fleeing, surrender, truce offers / replies, parley, bribes |
| `profiles.py` | `Profile` parameter sets (brute ... vip) |
| `heroes.py`, `explore.py` | hero bots (class profiles), exploration, loot, doors, rests |
| `vip.py`, `decide.py`, `random_policy.py`, `registry.py`, `arena.py` | escort NPC, reaction answers, random baseline, registry, test harness |

Per-creature memory lives in `Creature.known_ai_state` (a dict that is replaced, never mutated, so `clone/restore` are
safe); per-turn data is stored under key `turn` with a `(round, mode)` token. Tie-breaking uses a hash of
`(world.seed, round, eid)`, never RNG state. The only world mutation outside `step` is `update_awareness` and sharing
ally sightings into `c.aware`, plus the engine morale API (`morale_check`, `set_fleeing`).

## Scoring (expected hit points)
damage `min(E[dmg], hp)`; kill `P(kill)*(1.6*threat+2)`; focus `focus_w*(1-hp%)*threat`; casters/healers/leaders `caster_w`;
guard (`foes next to squishy allies`); isolation; control `P(fail)*cond_value*(1.4*threat+1.5)`; AoE = sum of per-target
values minus friendly fire (`friendly_fire_w` allies, 8 truce, 3 neutral), zones x1.8; healing `min(heal,missing)*urgency`,
reviving a dying ally x3; buffs table `BUFF_VALUE` x recipients; costs: spell slot `slot_w*(0.9+0.8*level)`, replacing
concentration `conc_hold*current value`. Movement gain = `S(best cell) - S(here)` with
`S = off + band + ally + flank + cover - safety*risk - opportunity attacks`; shoot first when a good shot exists.

## Behaviour notes
* Morale: check when hp < `creature.morale` (cowards at the first wound) and hp dropped, or half the faction fell;
  `world.morale_check` -> FLEE or SURRENDER (`can_surrender`). Fleeing: disengage if in reach, run for an exit marker
  (APPROACH + interact to escape), rally after two quiet turns, surrender when cornered. Surrendered: only end turn.
* Truce: accept if bribed >= `bribe_price` (10 gp x challenge per member), losing (power < 70 % or hp < 45 %), or the
  offerer is at least as strong and never hurt us; never from a faction that betrayed us; losing creatures with INT>=4 offer
  truce / parley.
* Doors: closed doors are opened with `interact` (key / thieves' tools / force for locked); monsters open the door towards
  known foes, heroes open doors when nothing else is left to explore.
* Heroes out of combat: heal, vote short rest, loot, objective regions (and the escort goal), frontier exploration with
  a leader (lock-picker first) and followers; stuck leaders make followers step aside.

## Adding a profile
1. `profiles.py`: `MYPROFILE = BRUTE.with_(name="mine", band=(4, 10), safety=0.8, ...)`; add it to `PROFILES`.
2. Anything special (new tactic): subclass `UtilityPolicy` (`profile_for`, `idle`, `fight`, `extra_candidates`, `goal`)
   and construct it in `registry._make_profile_policy`.
3. New ability kind: extend `abinfo._classify` (role) and add a handler to `utility._HANDLERS`.
4. Add the name to a test in `tests/test_ai.py`; run `python benchmarks/bench_ai.py`.

## Leaving the map, escorts (pass 2)
* `HeroPolicy.wants_escape`: when the scenario pays for escaping (`terminal_payoff` contains "escaped", an unfinished ESCAPE
  objective, or `relic_escape`) and no awake hostile is near, all non-escape objectives and loot are done (or the party
  is hurt below 35 % / 70 % of the rounds are gone), heroes walk to an exit marker and `interact`; a sealed exit sends
  them to crank the winch / pull the lever instead.
* Only helpless foes left: bots walk up and finish them (`finish_downed`).
* Escorts (`vip` tag ally): never advance more than 8 squares ahead of the VIP, attackers within 4 squares of the VIP get a
  large target bonus, escorts vacate the goal area so the VIP can enter; the VIP retreats toward escorts when threatened.
