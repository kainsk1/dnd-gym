"""Ranger: favored_enemy, natural_explorer, primeval_awareness, hunters_prey (colossus_slayer, giant_killer,
horde_breaker), fighting styles, extra_attack."""
import pytest

from emberdeep.core import resolution as R
from emberdeep.core.events import EventType as ET
from emberdeep.core.types import Ability, CreatureType, Skill
from features_fixtures import (
    begin, d, end_turn, events, has_ability, sandbox, script, set_ac, spawn_mon, spawn_pc, take_turn, usable, use,
)
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")


def setup(level=3, prey=None, mon="goblin", mpos=(6, 5), hp=300):
    w = sandbox(12)

    def mut(b):
        if prey:
            b.choices["hunters_prey"] = prey
    r = spawn_pc(w, "ranger", level, 5, 5, mutate=mut)
    m = spawn_mon(w, mon, *mpos)
    set_ac(w, m, 1)
    w.entities[m].hp = w.entities[m].max_hp = hp
    begin(w, r)
    return w, r, m


def test_favored_enemy_advantage_on_survival_and_lore_checks():
    w, r, m = setup(1)
    c = w.entities[r]
    undead = spawn_mon(w, "zombie", 9, 9)
    gob = w.entities[m]
    ctx = R.CheckCtx(c, Skill.SURVIVAL, target=w.entities[undead])
    for h in R.hooks(c, "check_mods"):
        h.check_mods(w, ctx)
    assert "favored_enemy" in ctx.adv
    ctx = R.CheckCtx(c, Skill.SURVIVAL, target=gob)
    for h in R.hooks(c, "check_mods"):
        h.check_mods(w, ctx)
    assert "favored_enemy" not in ctx.adv
    ctx = R.CheckCtx(c, Ability.INT, target=w.entities[undead])
    for h in R.hooks(c, "check_mods"):
        h.check_mods(w, ctx)
    assert "favored_enemy" in ctx.adv


def test_natural_explorer_registered_inert():
    from emberdeep.rules.registry import FEATURES
    assert "natural_explorer" in FEATURES


def test_primeval_awareness_spends_a_slot_and_reports_types():
    w, r, m = setup(3)
    c = w.entities[r]
    spawn_mon(w, "zombie", 15, 15)
    n = sum(c.spell_slots)
    use(w, r, "primeval_awareness")
    assert sum(c.spell_slots) == n - 1
    ev = [e for e in events(w, ET.ABILITY_USE) if e.data.get("ability") == "primeval_awareness_result"]
    assert ev and int(CreatureType.UNDEAD) in ev[0].data["found"]
    assert int(CreatureType.HUMANOID) not in ev[0].data["found"]


def test_colossus_slayer_once_per_turn_against_wounded():
    w, r, m = setup(3, hp=400)
    c = w.entities[r]
    gob = w.entities[m]
    gob.hp = gob.max_hp - 10                             # below its maximum
    script(w, d(15), d(4, 6), d(5, 8))
    use(w, r, "shortsword", m)
    dmg = events(w, ET.DAMAGE)
    first = sum(e.amount for e in dmg)
    assert len(dmg) == 2 and first == 4 + c.mods[1] + 5          # shortsword die + DEX + the extra 1d8
    c.ts.action_used = False                             # (Extra Attack stand-in) a second hit in the same turn
    c.ts.attack_committed = False
    script(w, d(15), d(4, 6), d(5, 8))
    use(w, r, "shortsword", m)
    second = sum(e.amount for e in events(w, ET.DAMAGE))
    assert second == 4 + c.mods[1]                        # once per turn only


def test_colossus_slayer_not_against_unwounded():
    w, r, m = setup(3, hp=400)
    script(w, d(15), d(4, 8), d(5, 8))
    use(w, r, "shortsword", m)
    dmg = sum(e.amount for e in events(w, ET.DAMAGE))
    assert dmg < 4 + 5 + 3                                   # no extra d8 against a full-health target


def test_giant_killer_reacts_to_a_large_attacker():
    w, r, m = setup(3, prey="giant_killer", mon="ogre", mpos=(6, 5))
    ogre = w.entities[m]
    c = w.entities[r]
    set_ac(w, m, 20)
    assert not any(a.id == "giant_killer" for a in w.abilities(r))
    atk = w.best_attack(ogre, c)
    events(w)
    script(w, d(20), d(5, 8), d(15), d(3, 8), default=0.4)
    atk.execute(w, ogre, r, None)
    ev = events(w)
    assert any(e.t == int(ET.REACTION) and e.data.get("ability") == "giant_killer" for e in ev)
    assert any(e.t == int(ET.ATTACK) and e.src == r for e in ev)
    assert not c.reaction_available


def test_giant_killer_ignores_medium_creatures():
    w, r, m = setup(3, prey="giant_killer", mon="goblin", mpos=(6, 5))
    gob = w.entities[m]
    c = w.entities[r]
    atk = w.best_attack(gob, c)
    set_ac(w, m, 20)
    script(w, d(20), d(5, 6))
    atk.execute(w, gob, r, None)
    assert not any(e.t == int(ET.REACTION) for e in events(w))


def test_horde_breaker_extra_attack_on_adjacent_enemy():
    w, r, m = setup(3, prey="horde_breaker", mpos=(6, 5))
    other = spawn_mon(w, "goblin", 6, 6)
    set_ac(w, other, 1)
    w.entities[other].hp = w.entities[other].max_hp = 300
    c = w.entities[r]
    script(w, d(15), d(3, 6), d(15), d(3, 6))
    use(w, r, "shortsword", m)
    atks = events(w, ET.ATTACK)
    assert len(atks) == 2 and {a.dst for a in atks} == {m, other}
    # the Attack action itself was charged only once
    assert c.ts.action_used
    # once per turn
    from features_fixtures import fresh_turn
    fresh_turn(w, r)
    c.ts.attack_committed = False
    script(w, d(15), d(3, 6), d(15), d(3, 6))
    use(w, r, "shortsword", m)
    assert len(events(w, ET.ATTACK)) == 1                  # no second bonus attack the same turn


def test_hunters_prey_default_is_colossus_slayer_and_container_idempotent():
    w, r, m = setup(3)
    ids = [h.ID for h in w.entities[r].hooks]
    assert ids.count("colossus_slayer") == 1
    w.abilities(r)
    assert [h.ID for h in w.entities[r].hooks].count("colossus_slayer") == 1


def test_ranger_extra_attack_at_5_and_archery():
    w, r, m = setup(5)
    c = w.entities[r]
    assert "archery" in {h.ID for h in c.hooks}
    for _ in range(2):
        script(w, d(15), d(3, 8))
        use(w, r, "longbow", m) if False else use(w, r, "shortsword", m)
    assert not usable(w, r, "shortsword")
