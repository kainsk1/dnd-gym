"""docs/ENV_API.md generated tables must match the code (run `python -m emberdeep.env.docgen --write docs/ENV_API.md`)."""
import os

from emberdeep.env import docgen
from emberdeep.env import spaces as SP

DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "ENV_API.md")


def test_generated_blocks_are_up_to_date():
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    assert docgen.update_markdown(text) == text, "docs/ENV_API.md is stale: run python -m emberdeep.env.docgen --write"
    for name in docgen.BLOCKS:
        assert f"<!-- BEGIN GENERATED:{name} -->" in text


def test_feature_sets_have_unique_names_and_docs():
    for fs in (SP.self_features(), SP.entity_features(8), SP.entity_features(0), SP.ability_features()):
        assert len(set(fs.names)) == len(fs.names) == len(fs.docs)
        assert all(d for d in fs.docs)
    assert len(SP.MAP_CHANNELS) == SP.N_MAP_CHANNELS == len({n for n, _ in SP.MAP_CHANNELS})
