"""Generates the reference tables of ``docs/ENV_API.md`` from the code that defines them.

    python -m emberdeep.env.docgen --write docs/ENV_API.md     # refresh the generated blocks in place
    python -m emberdeep.env.docgen                              # print all blocks to stdout

A generated block in the markdown file looks like ``<!-- BEGIN GENERATED:self_features -->`` ...
``<!-- END GENERATED:self_features -->``; everything between the markers is replaced.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys

from emberdeep.env import spaces as SP
from emberdeep.env.config import EnvConfig
from emberdeep.env.metrics import METRIC_DOCS
from emberdeep.env.rewards import COMPONENT_DOCS, COMPONENTS, DEFAULT_WEIGHTS


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def feature_table(fs: SP.FeatureSet) -> str:
    return _table(["idx", "name", "description"], [[i, f"`{n}`", d] for i, n, d in fs.rows()])


def block_self_features() -> str:
    fs = SP.self_features()
    return f"`self` has **{len(fs)}** features.\n\n" + feature_table(fs)


def block_entity_features() -> str:
    fs = SP.entity_features(8)
    return (f"`entities[slot, :]` has **{len(fs)}** features with the default `comm_vocab=8` "
            f"({len(SP.entity_features(0))} without the `said.*` block; the block grows with `comm_vocab`).\n\n"
            + feature_table(fs))


def block_ability_features() -> str:
    fs = SP.ability_features()
    return f"`abilities[k, :]` has **{len(fs)}** features.\n\n" + feature_table(fs)


def block_map_channels() -> str:
    return _table(["channel", "name", "description"], [[i, f"`{n}`", d] for i, (n, d) in enumerate(SP.MAP_CHANNELS)])


def block_state_layout() -> str:
    rows = [[i, f"`{n}`", d] for i, (n, d) in enumerate(SP.STATE_CREATURE_FIELDS)]
    t1 = _table(["field", "name", "description"], rows)
    ch = _table(["channel", "name"], [[i, f"`{n}`"] for i, n in enumerate(SP.STATE_MAP_CHANNELS)])
    return (f"Per creature slot ({len(SP.STATE_CREATURE_FIELDS)} fields; slots are the agents in `possible_agents` "
            f"order followed by every other creature by entity id; dead creatures keep their slot):\n\n{t1}\n\n"
            f"Map summary: {len(SP.STATE_MAP_CHANNELS)} channels downsampled (block average) to "
            f"{SP.STATE_GRID}x{SP.STATE_GRID}:\n\n{ch}\n\n"
            f"Scenario block ({SP.state_extra_dim()} values): round fraction, combat flag, env-step fraction, "
            f"fraction of agents alive, (done, failed) for up to {SP.MAX_STATE_OBJECTIVES} objectives, the "
            f"{SP.MAX_FACTIONS}x{SP.MAX_FACTIONS} faction relation matrix ((relation+1)/4), and the log-scaled "
            f"gold of each agent.")


def block_action_layout() -> str:
    cfg = EnvConfig()
    lay = SP.make_layout(cfg)
    rows = [
        ["`0`", "END_TURN (NOOP when you are not the decider)"],
        ["`1..8`", "MOVE N, NE, E, SE, S, SW, W, NW"],
        ["`9`, `10`", "ASCEND, DESCEND"],
        [f"`11 + k*T + t`", f"ABILITY `k` on target slot `t` (`k < K = {lay.K}`, `t < T = E + P = {lay.T}`)"],
        [f"`11 + K*T + e`", f"APPROACH entity slot `e` (`e < E = {lay.E}`)"],
        [f"`11 + K*T + E + v`", f"SAY token `v` (`v < V = {lay.V}`)"],
        [f"`11 + K*T + E + V + d`", f"DECIDE option `d` (`d < D = {lay.D}`)"],
    ]
    return (_table(["index", "action"], rows) +
            f"\n\nWith the defaults `A = 11 + K*T + E + V + D = {lay.A}` "
            f"(`K={lay.K}`, `E={lay.E}`, `P={lay.P}`, `T={lay.T}`, `V={lay.V}`, `D={lay.D}`).")


def block_reward_components() -> str:
    rows = [[f"`{c}`", DEFAULT_WEIGHTS.get(c, 0.0), COMPONENT_DOCS.get(c, "")] for c in COMPONENTS]
    return _table(["component", "default weight", "definition"], rows)


def block_metrics() -> str:
    return _table(["metric", "definition"], [[f"`{k}`", v] for k, v in METRIC_DOCS.items()])


def block_config() -> str:
    cfg = EnvConfig()
    rows = []
    for f in dataclasses.fields(EnvConfig):
        v = getattr(cfg, f.name)
        rows.append([f"`{f.name}`", f"`{v!r}`"])
    return _table(["field", "default"], rows)


BLOCKS = {
    "action_layout": block_action_layout,
    "self_features": block_self_features,
    "entity_features": block_entity_features,
    "ability_features": block_ability_features,
    "map_channels": block_map_channels,
    "state_layout": block_state_layout,
    "reward_components": block_reward_components,
    "metrics": block_metrics,
    "config": block_config,
}


def render_all() -> dict[str, str]:
    return {k: fn() for k, fn in BLOCKS.items()}


def update_markdown(text: str) -> str:
    """Replace every generated block of ``text`` with fresh content."""
    blocks = render_all()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN GENERATED:{name} -->)(.*?)(<!-- END GENERATED:{name} -->)", re.S)
        text, n = pat.subn(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m emberdeep.env.docgen")
    ap.add_argument("--write", metavar="FILE", help="refresh the generated blocks of this markdown file")
    a = ap.parse_args(argv)
    if a.write:
        with open(a.write, encoding="utf-8") as f:
            text = f.read()
        new = update_markdown(text)
        with open(a.write, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"updated {a.write}")
    else:
        for k, v in render_all().items():
            print(f"## {k}\n\n{v}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
