"""Episode recording and loading (``.edr`` files).

An ``.edr`` file is a gzip-compressed pickle of::

    {"format": "emberdeep-replay", "version": 1,
     "scene":  StaticScene,                         # once per episode
     "frames": [(FrameState, [event dict, ...])],   # one per env step (+ the initial frame)
     "meta":   {scenario, seed, agents, ...}}

— exactly the objects the renderer consumes, so a replay is played by pushing the frames to
``emberdeep.render.Renderer`` (see :mod:`emberdeep.viewer`).  Only trusted files should be loaded
(``pickle``).

Usage::

    env = emberdeep.make_aec("goblin_ambush")
    env = Recorder(env, "out.edr")      # works for AEC and Parallel envs (and the Gymnasium wrapper)
    env.reset(seed=0); ...              # play; the file is written when the episode ends / on close()
"""
from __future__ import annotations

import gzip
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

FORMAT = "emberdeep-replay"
VERSION = 1


@dataclass
class Replay:
    """A loaded replay: static scene, frames ``[(FrameState, events)]`` and metadata."""
    scene: Any
    frames: list
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)


def save_replay(path: str, scene, frames: list, meta: dict | None = None) -> str:
    """Write a replay file (creates parent directories)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    payload = {"format": FORMAT, "version": VERSION, "scene": scene, "frames": frames, "meta": dict(meta or {})}
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_replay(path: str) -> Replay:
    """Read a replay file written by :func:`save_replay` / :class:`Recorder`."""
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not an Emberdeep replay")
    if payload.get("version", 0) > VERSION:
        raise ValueError(f"{path} was written by a newer Emberdeep (replay version {payload['version']})")
    return Replay(payload["scene"], payload["frames"], payload.get("meta", {}))


def _unwrap_core(env):
    e = env
    for _ in range(8):
        core = getattr(e, "core", None)
        if core is not None:
            return core
        nxt = getattr(e, "unwrapped", None)
        if nxt is None or nxt is e:
            break
        e = nxt
    raise TypeError("Recorder needs an Emberdeep environment (AEC, Parallel or Gym)")


class Recorder:
    """Transparent wrapper recording every step of an Emberdeep env.

    path       file to write (``None`` keeps episodes in memory only).  The first episode goes to ``path``;
               later ones to ``<stem>_<n>.edr``.
    keep       keep finished episodes in ``self.episodes`` (list of :class:`Replay`).
    """

    def __init__(self, env, path: str | None = None, keep: bool = False):
        self.env = env
        self.path = path
        self.keep = keep
        self.core = _unwrap_core(env)
        from emberdeep.env.render_adapter import FrameBuilder
        self.builder = FrameBuilder(self.core)
        self.episodes: list[Replay] = []
        self.files: list[str] = []
        self._frames: list = []
        self._scene = None
        self._n_episodes = 0
        self._last_outcome = None
        self.annotations: dict = {}

    # ------------------------------------------------------------------------------------ delegation
    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    # ------------------------------------------------------------------------------------ recording
    def _begin(self) -> None:
        self._flush()
        self.builder.reset()
        self._scene = self.builder.scene()
        ev = list(self.core.reset_events)
        frame = self.builder.frame(ev, None, self.annotations)
        self._frames = [(frame, self.builder.event_dicts(ev))]
        self._last_outcome = self.core.last_outcome

    def _record(self) -> None:
        out = self.core.last_outcome
        if out is None or out is self._last_outcome:
            return                                  # dead step or no new engine step
        self._last_outcome = out
        frame = self.builder.frame(out.events, out, self.annotations)
        self._frames.append((frame, self.builder.event_dicts(out.events)))
        if self.core.done:
            self._flush()

    def _meta(self) -> dict:
        w = self.core.world
        return {"scenario": w.name, "seed": int(self.core.last_seed or 0), "agents": list(self.core.agents),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": len(self._frames) - 1,
                "outcome": dict(w.outcome) if w.done else {}, "version": VERSION}

    def _flush(self) -> None:
        if not self._frames or self._scene is None:
            return
        rep = Replay(self._scene, self._frames, self._meta())
        if self.keep:
            self.episodes.append(rep)
        if self.path:
            if self._n_episodes == 0:
                target = self.path
            else:
                stem, ext = os.path.splitext(self.path)
                target = f"{stem}_{self._n_episodes}{ext or '.edr'}"
            save_replay(target, rep.scene, rep.frames, rep.meta)
            self.files.append(target)
        self._n_episodes += 1
        self._frames = []

    def set_annotations(self, annotations: dict) -> None:
        """Research overlay (``{agent: {"value", "top_actions", "note"}}``) stored in subsequent frames."""
        self.annotations = dict(annotations)
        if hasattr(self.env, "set_annotations"):
            self.env.set_annotations(annotations)

    # ------------------------------------------------------------------------------------ API passthrough
    def reset(self, *args, **kw):
        res = self.env.reset(*args, **kw)
        self._begin()
        return res

    def step(self, *args, **kw):
        res = self.env.step(*args, **kw)
        self._record()
        return res

    def close(self) -> None:
        self._flush()
        if hasattr(self.env, "close"):
            self.env.close()

    def save(self, path: str | None = None) -> str | None:
        """Write the episode in progress now (recording continues) and return the file name."""
        if not self._frames or self._scene is None:
            return None
        p = path or self.path
        if p is None:
            raise ValueError("no path given")
        return save_replay(p, self._scene, list(self._frames), self._meta())
