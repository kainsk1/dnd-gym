# Changelog

## 1.1.0 — fixes for the v1.0 limitations

### Rules (SRD 5.1)
* **Multiclassing**: `CharacterBuild.classes`, ability-score prerequisites, multiclass proficiencies, HP per class
  hit die, combined spell slots (multiclass spellcaster table; Pact Magic separate), per-class spellcasting ability,
  non-stacking Extra Attack / Unarmored Defense, multiclass archetypes (`paladin_sorcerer`, `fighter_wizard`,
  `rogue_fighter`, `cleric_druid`) and `random_build(multiclass_prob=...)`.
* **Carrying capacity & encumbrance** (`RulesConfig.encumbrance`: `off` | `capacity` (default) | `variant`).
* **Attunement**: 3-item limit, attuned at spawn and short rests, `World.attune/unattune`.
* **Mounted combat**: mount/dismount, controlled & independent mounts, riders share the mount's square, forced
  dismount saves, opportunity attacks choose rider or mount; new scenario `mounted_patrol`.
* **Movement**: climbable low walls/crates (extra cost, climb speed), burrowing (z = −1, tremorsense), swimming and
  **underwater combat**, **squeezing** (Large+ through narrow gaps), **long jumps** over pits (Jump spell ×3),
  incorporeal movement through walls/objects.
* **Other**: stable creatures regain 1 HP after 1d4 hours (rests), Wind Wall / gas barriers block projectiles and
  gas, sealed exits block fliers, per-observer Minor Illusion, gaze attacks with avert-eyes decisions.
* **Class features**: all metamagic exact (Heightened, Careful/Sculpt incl. conditions, Distant, Subtle, Quickened
  after the Action, Twinned), Bardic Inspiration & Cutting Words as post-roll decisions (attacks, saves, checks,
  contests, damage), Song of Rest per Hit Die, Lay on Hands amount choice, monk weapons use DEX, Halfling
  Nimbleness, Devil's Sight, Dark One's Blessing on drops to 0 HP, Rage blocks spellcasting, Danger Sense.
* **Spells**: SPELL_CHOICE decisions offered to agent casters (command, blindness/deafness, enhance ability,
  enlarge/reduce, protection from energy, bestow curse, calm emotions, suggestion, conjure animals, animate dead,
  lesser restoration), Slow (action XOR bonus), Sanctuary vs targeted spells, Charm Person, Dispel Magic on
  non-concentration conditions, Spiritual Weapon measured from the weapon, Web burns.
* **Monsters**: lycanthrope form restrictions (changing form costs an action), Mummy Rot as a curse,
  sleep poison no longer double-poisons, fire elemental / gelatinous cube enter spaces, swarm & incorporeal
  space sharing.
* **Correctness**: `World.estimate` is now pure (no RNG use — Sanctuary/Mirror Image previously rolled during
  AI estimates), Large-creature path costs use the worst square of the footprint, free abilities are once per
  turn plus `RulesConfig.max_actions_per_turn` (no stalling exploit), reset never returns an already-finished
  episode.

### Performance
* AEC agent-steps/s per core on `goblin_ambush`: ~630 → ~1,000 (compiled observation kernels, memoised
  masks/path fields, economy-gated ability masks, batched estimates for the scripted AI). Observations, masks
  and trajectories are bit-identical to the reference implementations (differential tests).
* HUD draw: ~3.7 ms → ~1.8 ms per animated frame (per-panel caches, dirty tracking); idle frames ~0.6 ms.
* Scripted AI ~1.4× faster per decision.

### Scripted baselines
* Every scenario ends by objective / escape / defeat in ≥ 8/10 seeds with all-scripted play (most 10/10).

### Observation changes (regenerate any hard-coded feature indices)
* `self` gained decision-kind one-hots for `SPELL_CHOICE`, `ON_FALL`, `POST_ROLL`, and status bits for the new
  statuses (mounted, squeezing, burrowed...). See `docs/ENV_API.md` (generated from code).

## 1.0.0 — first release
