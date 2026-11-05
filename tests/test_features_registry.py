"""Registry completeness, documentation coverage and the factory-handled marker features."""
import os
import re

import pytest

from emberdeep.core.types import Ability
from emberdeep.data.archetypes import archetype
from emberdeep.data.classes import CLASSES, FEATURE_HOOKS_REFERENCED, LAND_CIRCLE_SPELLS
from emberdeep.data.races import RACE_TRAIT_IDS
from emberdeep.rules.registry import FEATURES, TRAITS
from features_fixtures import begin, sandbox, spawn_mon, spawn_pc
from features_fixtures import no_swallowed_errors  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_swallowed_errors")

DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "FEATURES.md")


def test_every_referenced_feature_and_trait_is_registered():
    assert not sorted(i for i in FEATURE_HOOKS_REFERENCED if i not in FEATURES)
    assert not sorted(i for i in RACE_TRAIT_IDS if i not in TRAITS)


def test_features_doc_covers_every_id():
    text = open(DOC, encoding="utf-8").read()
    missing = [i for i in sorted(FEATURE_HOOKS_REFERENCED | RACE_TRAIT_IDS) if f"`{i}`" not in text
               and i not in ("ability_score_improvement", "subclass_marker")]
    # subclass choice markers (primal_path, bard_college ...) share the `subclass_marker` hook
    assert not missing, missing
    assert "`ability_score_improvement`" in text and "`subclass_marker`" in text


def test_asi_applied_by_the_factory_and_marker_hooks_present():
    w = sandbox(1)
    f = spawn_pc(w, "fighter", 4, 5, 5)
    spawn_mon(w, "goblin", 15, 15)
    b = w.entities[f].build
    assert b.asi
    base = list(b.base_abilities)
    from emberdeep.data.races import RACES
    for a, v in RACES[b.race].ability_bonuses.items():
        base[int(a)] += v
    for a in b.asi:
        base[int(a)] += 1
    assert w.entities[f].scores == base
    ids = {h.ID for h in w.entities[f].hooks}
    assert "ability_score_improvement" in ids and "subclass_marker" in ids


def test_domain_oath_and_circle_spells_are_prepared():
    w = sandbox(1)
    cl = spawn_pc(w, "cleric", 5, 5, 5)
    pal = spawn_pc(w, "paladin", 5, 6, 6)
    dr = spawn_pc(w, "druid", 5, 7, 7)
    c, p, d = (w.entities[i] for i in (cl, pal, dr))
    assert {"bless", "cure_wounds", "lesser_restoration", "spiritual_weapon", "beacon_of_hope", "revivify"} <= set(c.spells)
    assert {"protection_from_evil_and_good", "sanctuary"} <= set(p.spells)
    forest = set(LAND_CIRCLE_SPELLS["forest"][3]) | set(LAND_CIRCLE_SPELLS["forest"][5])
    assert forest <= set(d.spells)


def test_every_class_gets_all_its_registered_features_attached():
    for cls in CLASSES:
        w = sandbox(1)
        e = spawn_pc(w, cls, 5, 5, 5)
        spawn_mon(w, "goblin", 15, 15)
        begin(w, e)
        w.abilities(e)
        have = {type(h) for h in w.entities[e].hooks}
        from emberdeep.core.factory import lookup_feature
        for fid in w.entities[e].feature_ids:
            fd = lookup_feature(fid)
            if fd is not None:
                assert FEATURES[fd.hook or fd.id] in have, (cls, fid)
