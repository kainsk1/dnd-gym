# Scenario Library — Game-Theoretic Design

Each scenario is a *substrate* (in the Melting Pot sense): a fixed physical/rules setting whose
payoff structure creates a specific strategic tension. All scenarios expose the same observation
and action spaces, so policies, populations and evaluation protocols transfer across them.

Notation: `N` = number of RL agents. "Individual payoff" means the agent's own reward stream
before social mixing. Every scenario lists (a) the tension, (b) the mechanics that create it,
(c) default reward weights, (d) what cooperative vs. selfish play looks like, (e) metrics to
watch. Numbers are defaults — scenario kwargs may override them (see `ScenarioSpec.randomize`).

Shared defaults (env/rewards.py `DEFAULT_WEIGHTS`): victory +1.0, defeat −1.0, death −0.5,
downed −0.1, kill_xp 0.0005/XP, damage_dealt 0.004/hp, damage_taken −0.004/hp,
friendly_damage −0.01/hp, healing_given 0.004/hp, revives +0.2, gold 0.001/gp, objective +1.0,
time −0.0005/round, betrayal 0.0 (researchers price it), spared 0.0.

---------------------------------------------------------------------------------------------

## 1. `goblin_ambush` — pure cooperation baseline
* Map `goblin_ambush` (forest road). Party: fighter, cleric, rogue, wizard (level 2, agents).
  Goblins hidden in brush (`a,c,d,e`, Stealth-rolled, tag `hidden`), goblin archers on the
  flank (`f`), a bugbear leader (`g`). Start mode: exploration (the party walks into it).
* Objective: DEFEAT_ALL hostile (fled/surrendered count). Victory ends the episode.
* Tension: none beyond credit assignment and coordination (focus fire, protect the wizard,
  heal the downed). The sanity-check scenario: a good algorithm should reach >90% win rate.
* Variants: `goblin_ambush_small` (2 PCs vs 3 goblins, level 1, 20×14 crop) for fast debugging;
  `goblin_ambush_team` (`team_reward=True`).

## 2. `dragon_hoard` — stag hunt
* Map `dragon_hoard`. Party of 4 level-4 PCs (agents). A red dragon wyrmling (`d`, tag `asleep`
  until someone enters `dragon_chamber` or makes noise in it) guards the hoard: 3 chests + 8 loot
  piles ≈ 3,000 gp total. Two alcove chests (≈ 150 gp each) guarded by 3 kobolds each (`k`,`j`).
  Kobold sentries `m`, chamber guards `g`.
* Payoff: `terminal_payoff = "gold_if_escaped"` (personal gold × 0.001 counted only if the agent
  leaves through `E` alive) + survival. No victory bonus.
* Tension (stag hunt): the dragon (CR 4, ~75 HP, 7d10 fire breath) kills 1–2 heroes who engage
  alone, but 4 coordinated heroes beat it with high probability. Hare = loot an alcove chest and
  escape (safe ≈ 150 gp). Stag = all fight the dragon (≈ 750 gp each if shared evenly).
  Secondary tension after the stag hunt: hoard piles are personal pickups (division of spoils;
  `give_gold` can equalise).
* Metrics: fraction of episodes where all agents engage the dragon, dragon kill rate, Gini of
  escaped gold, solo-alcove rate.

## 3. `cursed_coffer` — prisoner's dilemma / chicken over a single prize
* Map `cursed_coffer`. Two factions of 2 agents each (`west`: markers 0–3? use 0–1; `east`: 4–5),
  relation NEUTRAL. One coffer (`C`) with 1,000 gp. Two animated-armor guardians (`g`,`h`)
  activate when the coffer is opened, and **preferentially attack whoever carries cursed gold**
  (items `cursed_coffer_gold` worth 100 gp each, 10 of them; carriers get effect `cursed_marker`).
* Each faction exits via its own entrance. Payoff: personal gold if escaped; death −0.5.
* Tension: both factions truce and split (cooperate) → guardians are beaten together, each
  agent ≈ 250 gp; one faction attacks while the other loots (defect) → temptation payoff if it
  works, but the carrier is hunted by guardians; mutual fighting → guardians + PvP losses (worst).
  Truce/betrayal actions and `give_gold` make agreements expressible; SAY tokens allow cheap talk.
* Metrics: truce formation rate, betrayal rate, gold split equality across factions, PvP damage.

## 4. `rival_delvers` — team competition with optional alliance
* Map `rival_delvers` (symmetric). Two parties of 3 agents (`west`, `east`), relation NEUTRAL.
  Monster rooms between (`a,c,d,e`), central relic room guarded by an ogre + 2 orcs (`r`).
  The `ember_relic` (1,000 gp) lies on an altar in `relic_room`.
* Objective: COLLECT `ember_relic` and ESCAPE through the party's own exit (`exit_west`/`exit_east`
  regions). The holder's faction gets `victory`; the other `defeat`. Gold found en route is personal.
* Tension: guardians are too strong for one party but beatable by both; after they fall the
  parties compete for the relic. Truce → cooperate vs guardians → race/betray. Zero-sum at the end,
  positive-sum before — classic "coopetition".
* Metrics: alliance formation, timing of betrayal (rounds after guardians fall), relic win rate per
  faction, total monsters killed by both.

## 5. `fountain_of_life` — common-pool resource
* Map `fountain_of_life`. Party of 4 level-3 PCs. The central altar `A` is a **Fountain of Life**
  (`rules/objects.py` hook `fountain`): `charges` start 6, capacity 8; `interact` (ACTION) drinks one
  charge: heal 2d8 + 2. At the start of every round charges regrow logistically:
  `c ← c + r·c·(1 − c/K)` with r = 0.25, accumulated fractionally (displayed as an integer).
  At c = 0 it never regrows (tragedy).
* Waves (`WaveSpec`): goblins/skeletons from the four entrances (`a,c,d,e`) every 3 rounds,
  escalating (`scale_with_round`), for 18 rounds. Objective: SURVIVE 18 rounds (victory) — at
  least one agent alive.
* Individual payoff: survival, damage dealt, kills; the fountain is the only healing besides the
  cleric's slots. Each agent is better off drinking when hurt; collectively, keeping c near K/2
  maximises total healing (maximum sustainable yield).
* Metrics: fountain depletion time, mean charge level, total healing from fountain, survival, Gini
  of fountain use.

## 6. `shrine_of_wards` — threshold public good
* Map `shrine_of_wards`. Party of 4 level-3 PCs, each starting with 100 gp. Altar `A` = Shrine
  (`rules/objects.py` hook `donation_shrine`): `interact` donates `give_gold_chunk` (25 gp) per use.
  If total donations ≥ 200 gp **before round 6**, the Ward activates: every party member gains effect
  `ward_of_the_shrine` (+2 AC, resistance to necrotic) for the rest of the episode.
* From round 6, undead waves (skeletons, zombies, a wight at round 12) attack from the chapels.
* Payoff: `terminal_payoff = "gold_if_alive"` (personal remaining gold) + survival + victory
  (SURVIVE 16 rounds). Free-rider incentive: let the others donate.
* Metrics: donation distribution, ward activation rate, free-riding index, survival.

## 7. `hold_the_bridge` — volunteer's dilemma
* Map `hold_the_bridge`. Party of 4 level-3 PCs carrying treasure (each 200 gp). Enemies (`a,c,d,f`
  hobgoblins, `g` archers, `k` captain) arrive across a 1-wide bridge over a chasm in waves.
  The exit `E` is a portcullis opened by a **winch** (altar hook `winch`): it requires 3 cumulative
  `interact` actions (any agents); once open, agents may escape.
* Payoff: `gold_if_escaped` + survival. Whoever holds the bridge chokepoint protects the others
  but is unlikely to escape in time. If nobody holds, enemies flood the exit area.
* Metrics: who volunteers (class, HP), escape count, holder death rate, ordering of escapes.

## 8. `gladiator_pit` — free-for-all with alliances
* Map `gladiator_pit`. 6 agents, each in its own faction (all HOSTILE), random classes level 3.
  At round 5 a manticore is released (wave from the gates) that attacks everyone.
* Payoff: rounds survived × 0.02, last-standing faction `victory` (+1); prize 1,000 gp to the
  winner(s) (allied survivors split it).
* Tension: temporary alliances (offer/accept truce) vs. betrayal; the shared threat forces
  short-lived cooperation; kingmaking.
* Metrics: truces formed/broken, betrayals, alliance duration, winner distribution.

## 9. `caravan_escort` — protection public good + bargaining
* Map `caravan_escort`. 4 agents escort a merchant (`v`, scripted NPC, tag `vip`, slow) to the exit.
  Bandits on ledges (`a,c,d,e,f,g`), a toll line (`h`) and a bandit captain (`k`) who can be
  `parley`ed or bribed: `give_gold` to the captain ≥ 100 gp total → bandits accept a TRUCE.
* Payoff: if the VIP reaches the exit alive, each surviving agent receives 150 gp (escort fee);
  bribes come out of the payer's personal purse. Kill XP for bandits.
* Tension: who pays the toll (public good with private cost); fight vs pay; protecting the VIP
  vs personal safety.
* Metrics: bribe contributions per agent, VIP survival, fights avoided.

## 10. `deep_delve` — procedural dungeon crawl (generalisation)
* Procedural map per episode (`bsp_dungeon` | `cave` | `crypt`, seed from env RNG). Party of 4
  random-class PCs (`random_build`, level 1–5). Encounters per room from `data/encounters.py`
  (`build_encounter` with party levels, difficulty medium–hard, themed by map), traps, locked
  doors with keys, chests (`roll_loot` tables), a boss room and an exit.
* Exploration mode first; short rests via unanimous `rest` votes; personal loot.
* Payoff: gold_if_escaped + kills + survival + objective (boss slain). Mixed motive throughout.

## 11. Micro scenarios (fast iteration)
* `duel` — 1v1 random classes level 3 in a small arena (two learning factions; zero-sum).
* `skirmish_2v2` — two learning teams of 2 (team competition).
* `goblin_ambush_small` — 2 PCs vs 3 goblins.

---------------------------------------------------------------------------------------------

## Scenario objects to implement (rules/objects.py)

| hook | params | behaviour |
|---|---|---|
| `fountain` | charges, capacity, regen_r, heal | interact (ACTION): heal; logistic regrowth each round; INTERACT event data {charges} |
| `donation_shrine` | threshold, deadline_round, chunk, effect | interact: donate chunk gp; activation applies effect to all faction members; GOLD_CHANGE events |
| `winch` | required, opens (door/exit eids or region) | interact: progress +1 (persisted); at required → opens portcullis door / enables exit |
| `relic_pedestal` | item, on_take (wave id / awaken tags) | interact: take relic item; triggers waves with trigger_item |
| `cursed_coffer` | items, guardian_tag | chest-like; items carry `cursed_marker` effect while held |

## Scripted behaviours needed (ai)
* `asleep` tag: unaware until noise/damage/perception; `guard:<region>` holds position.
* Guardians target cursed carriers; VIP NPC policy (`vip`): moves toward exit along road, stops
  when enemies adjacent, follows nearest escort.
* Bandit captain accepts bribe/parley thresholds (truce with the payer's faction).
