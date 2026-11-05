"""Vectorised Emberdeep environments with a batched parallel API.

``make_vector(scenario, num_envs=8)`` returns an :class:`AsyncVectorEnv`: one worker subprocess per environment
(``spawn`` context, so it is safe on every platform), observations written by the workers straight into
``multiprocessing.shared_memory`` buffers, actions / rewards / flags exchanged through shared arrays, and only a
tiny message per step over a pipe.  :class:`SyncVectorEnv` runs the same code in-process for debugging.

Batched API (``A`` = number of agents, identical in every sub-environment)::

    obs, infos = venv.reset(seed=0)
    obs         dict key -> ndarray [num_envs, A, ...]        (same keys as the single-agent observation)
    obs, rewards, terminations, truncations, infos = venv.step(actions)       # actions int [num_envs, A]
    rewards     float32 [num_envs, A]     mixed rewards of the step (0 for agents that were already dead)
    terminations bool   [num_envs, A]     agent died / escaped, or the episode ended; stays True until the reset
    truncations  bool   [num_envs, A]     episode truncated (max rounds / steps)
    infos       list of dict (one per env): ``acting`` bool [A], ``alive`` bool [A], ``seed``, and at the end of an
                episode ``episode`` (flat metrics per agent id + global), ``final_observation`` (observation
                of the last state, before the automatic reset) and ``episode_done=True``.

Environments reset automatically: when an episode ends, the returned ``obs`` already belong to the next episode
(its seed is drawn from the environment's own deterministic seed stream).  Only the decider's action of each
sub-environment is applied; the other agents' entries are ignored (their masks allow only NOOP = 0).
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import traceback
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from emberdeep.env.config import EnvConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------- runner
class _Runner:
    """One sub-environment writing into externally provided buffer rows (shared by sync and async)."""

    def __init__(self, cfg: EnvConfig):
        from emberdeep.env.core import EmberdeepCore
        self.core = EmberdeepCore(cfg)
        self.n = self.core.n_agents
        self.keys = list(self.core.observation_space.spaces.keys())
        self.alive_prev = np.ones(self.n, bool)

    def specs(self) -> dict:
        """``{key: (shape, dtype)}`` of one agent's observation."""
        return {k: (s.shape, s.dtype) for k, s in self.core.observation_space.spaces.items()}

    def _write_obs(self, bufs: dict) -> None:
        core = self.core
        for ai, a in enumerate(core.agents):
            o = core.observe(a)
            for k in self.keys:
                bufs[k][ai] = o[k]

    def reset(self, seed, bufs: dict) -> dict:
        self.core.reset(seed)
        self.alive_prev = np.ones(self.n, bool)
        self._write_obs(bufs)
        return self._info(None)

    def _info(self, out, final=None, ended=False) -> dict:
        core = self.core
        dec = core.decider_agent()
        info: dict[str, Any] = {
            "acting": [a == dec for a in core.agents],
            "alive": [core.is_live(a) for a in core.agents],
            "seed": core.last_seed,
        }
        if ended:
            info["episode_done"] = True
        return info

    def step(self, actions: np.ndarray, bufs: dict, rew: np.ndarray, term: np.ndarray, trunc: np.ndarray) -> dict:
        core = self.core
        dec = core.decider_agent()
        act = 0
        if dec is not None:
            act = int(actions[core.agent_index[dec]])
        was_alive = self.alive_prev.copy()
        out = core.step(dec, act)
        done, truncated = core.done, core.truncated
        alive = np.array([core.is_live(a) for a in core.agents], bool)
        rew[:] = np.where(was_alive, out.rewards, 0.0)
        dead = ~alive
        term[:] = dead | (done and not truncated)
        trunc[:] = (done and truncated) & ~dead
        self.alive_prev = alive
        info = self._info(out, ended=done)
        if done:
            self._write_obs(bufs)
            info["final_observation"] = {k: np.stack([bufs[k][i] for i in range(self.n)]) for k in self.keys}
            m = core.episode_metrics()
            info["episode"] = {a: {**m["agents"][a], **m["global"],
                                   **{"scenario/" + k: v for k, v in m["scenario"].items()}} for a in core.agents}
            core.reset(None)
            self.alive_prev = np.ones(self.n, bool)
            info["acting"] = [a == core.decider_agent() for a in core.agents]
            info["alive"] = [True] * self.n
            info["seed"] = core.last_seed
        self._write_obs(bufs)
        return info

    def call(self, name: str, *args):
        return getattr(self.core, name)(*args)


# ---------------------------------------------------------------------------------------------- sync
class _VecBase:
    """Shared API surface of the sync / async implementations."""

    num_envs: int
    n_agents: int
    agents: list[str]
    single_observation_space: Any
    single_action_space: Any

    def __len__(self) -> int:
        return self.num_envs

    @property
    def action_space(self):
        from gymnasium import spaces
        return spaces.MultiDiscrete(np.full((self.num_envs, self.n_agents), self.single_action_space.n))

    def seeds(self, seed: int | None) -> list[int | None]:
        if seed is None:
            return [None] * self.num_envs
        return [int(seed) + i for i in range(self.num_envs)]


class SyncVectorEnv(_VecBase):
    """In-process vector environment (same API as :class:`AsyncVectorEnv`; for debugging and profiling)."""

    def __init__(self, scenario: Any = "goblin_ambush", num_envs: int = 4, seed: int | None = None, **cfg: Any):
        base = EnvConfig.make(None, scenario=scenario, **cfg)
        self.num_envs = num_envs
        self.runners = [_Runner(base.replace(seed=(None if seed is None else int(seed) + i)))
                        for i in range(num_envs)]
        r0 = self.runners[0]
        self.n_agents = r0.n
        self.agents = list(r0.core.agents)
        self.single_observation_space = r0.core.observation_space
        self.single_action_space = r0.core.action_space
        self.state_space = r0.core.state_space
        self._specs = r0.specs()
        self.obs = {k: np.zeros((num_envs, self.n_agents) + sh, dt) for k, (sh, dt) in self._specs.items()}
        self.rewards = np.zeros((num_envs, self.n_agents), np.float32)
        self.terminations = np.zeros((num_envs, self.n_agents), bool)
        self.truncations = np.zeros((num_envs, self.n_agents), bool)
        self.closed = False

    def _bufs(self, i: int) -> dict:
        return {k: v[i] for k, v in self.obs.items()}

    def reset(self, seed: int | None = None):
        infos = [r.reset(s, self._bufs(i)) for i, (r, s) in enumerate(zip(self.runners, self.seeds(seed)))]
        self.terminations[:] = False
        self.truncations[:] = False
        self.rewards[:] = 0.0
        return self.obs, infos

    def step(self, actions):
        actions = np.asarray(actions)
        infos = [r.step(actions[i], self._bufs(i), self.rewards[i], self.terminations[i], self.truncations[i])
                 for i, r in enumerate(self.runners)]
        return self.obs, self.rewards, self.terminations, self.truncations, infos

    def state(self) -> np.ndarray:
        """Global CTDE state ``[num_envs, S]``."""
        return np.stack([r.core.state() for r in self.runners])

    def call(self, name: str, *args) -> list:
        return [r.call(name, *args) for r in self.runners]

    def episode_metrics(self) -> list[dict]:
        return self.call("episode_metrics")

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------------------------- async
def _attach(names: dict, specs: dict, num_envs: int, n_agents: int, i: int):
    """Open the shared buffers inside a worker and return row views for env ``i`` (plus the handles)."""
    handles = []
    views: dict = {}

    def view(shm_name, shape, dtype):
        shm = shared_memory.SharedMemory(name=shm_name)
        handles.append(shm)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr

    bufs = {k: view(names["obs:" + k], (num_envs, n_agents) + sh, dt)[i] for k, (sh, dt) in specs.items()}
    rew = view(names["rew"], (num_envs, n_agents), np.float32)[i]
    term = view(names["term"], (num_envs, n_agents), np.bool_)[i]
    trunc = view(names["trunc"], (num_envs, n_agents), np.bool_)[i]
    act = view(names["act"], (num_envs, n_agents), np.int64)[i]
    return bufs, rew, term, trunc, act, handles


def _worker(index: int, cfg: EnvConfig, conn, names: dict, specs: dict, num_envs: int, n_agents: int) -> None:
    """Subprocess main loop (top-level so it can be pickled under ``spawn``)."""
    handles = []
    try:
        bufs, rew, term, trunc, act, handles = _attach(names, specs, num_envs, n_agents, index)
        runner = _Runner(cfg)
        conn.send(("ready", None))
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                conn.send(("ok", runner.reset(payload, bufs)))
            elif cmd == "step":
                conn.send(("ok", runner.step(act, bufs, rew, term, trunc)))
            elif cmd == "call":
                name, args = payload
                conn.send(("ok", runner.call(name, *args)))
            elif cmd == "close":
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:  # noqa: BLE001
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass
        conn.close()


class AsyncVectorEnv(_VecBase):
    """Subprocess vector environment with shared-memory buffers (see the module docstring)."""

    def __init__(self, scenario: Any = "goblin_ambush", num_envs: int = 4, seed: int | None = None,
                 start_method: str = "spawn", timeout: float = 300.0, **cfg: Any):
        base = EnvConfig.make(None, scenario=scenario, **cfg)
        self.num_envs = num_envs
        self.timeout = timeout
        self.closed = True
        # probe once in the parent for spaces / agent list
        probe = _Runner(base.replace(seed=None if seed is None else int(seed)))
        self.n_agents = probe.n
        self.agents = list(probe.core.agents)
        self.single_observation_space = probe.core.observation_space
        self.single_action_space = probe.core.action_space
        self.state_space = probe.core.state_space
        self._specs = probe.specs()
        del probe
        n = self.n_agents
        self._shms: list[shared_memory.SharedMemory] = []
        names: dict[str, str] = {}

        def alloc(key: str, shape: tuple, dtype) -> np.ndarray:
            size = max(1, int(np.prod(shape)) * np.dtype(dtype).itemsize)
            shm = shared_memory.SharedMemory(create=True, size=size)
            self._shms.append(shm)
            names[key] = shm.name
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            arr[...] = 0
            return arr

        self.obs = {k: alloc("obs:" + k, (num_envs, n) + sh, dt) for k, (sh, dt) in self._specs.items()}
        self.rewards = alloc("rew", (num_envs, n), np.float32)
        self.terminations = alloc("term", (num_envs, n), np.bool_)
        self.truncations = alloc("trunc", (num_envs, n), np.bool_)
        self._actions = alloc("act", (num_envs, n), np.int64)
        ctx = mp.get_context(start_method)
        self._conns = []
        self._procs = []
        try:
            for i in range(num_envs):
                parent, child = ctx.Pipe()
                cfg_i = base.replace(seed=None if seed is None else int(seed) + i)
                p = ctx.Process(target=_worker, args=(i, cfg_i, child, names, self._specs, num_envs, n), daemon=True)
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)
            self.closed = False
            for i, c in enumerate(self._conns):
                self._expect(c, i)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    # ------------------------------------------------------------------------------------ plumbing
    def _expect(self, conn, i: int):
        if not conn.poll(self.timeout):
            raise TimeoutError(f"vector worker {i} did not answer within {self.timeout}s")
        status, payload = conn.recv()
        if status == "error":
            raise RuntimeError(f"vector worker {i} failed:\n{payload}")
        return payload

    def _broadcast(self, cmd: str, payloads=None) -> list:
        if self.closed:
            raise RuntimeError("vector environment is closed")
        for i, c in enumerate(self._conns):
            c.send((cmd, None if payloads is None else payloads[i]))
        return [self._expect(c, i) for i, c in enumerate(self._conns)]

    # ------------------------------------------------------------------------------------ API
    def reset(self, seed: int | None = None):
        infos = self._broadcast("reset", self.seeds(seed))
        self.terminations[:] = False
        self.truncations[:] = False
        self.rewards[:] = 0.0
        return self.obs, infos

    def step(self, actions):
        a = np.asarray(actions)
        if a.shape != (self.num_envs, self.n_agents):
            raise ValueError(f"actions must have shape ({self.num_envs}, {self.n_agents}), got {a.shape}")
        self._actions[...] = a
        infos = self._broadcast("step")
        return self.obs, self.rewards, self.terminations, self.truncations, infos

    def call(self, name: str, *args) -> list:
        """Call ``EmberdeepCore.<name>(*args)`` in every worker and return the results."""
        return self._broadcast("call", [(name, args)] * self.num_envs)

    def state(self) -> np.ndarray:
        return np.stack(self.call("state"))

    def episode_metrics(self) -> list[dict]:
        return self.call("episode_metrics")

    def close(self) -> None:
        if getattr(self, "closed", True):
            return
        self.closed = True
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:  # noqa: BLE001
                pass
        for c in self._conns:
            try:
                if c.poll(2.0):
                    c.recv()
            except Exception:  # noqa: BLE001
                pass
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        for p in self._procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        for shm in self._shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:  # noqa: BLE001
                pass
        self._shms = []
        self.obs = {}

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def make_vector_env(scenario: Any = "goblin_ambush", num_envs: int = 4, api: str = "parallel", sync: bool = False,
                    seed: int | None = None, **cfg: Any):
    """Factory behind :func:`emberdeep.make_vector`."""
    if api != "parallel":
        raise ValueError("make_vector supports api='parallel' (batched [num_envs, n_agents, ...] arrays); "
                         "use make_gym / make_aec for the other APIs")
    if sync:
        return SyncVectorEnv(scenario, num_envs, seed=seed, **cfg)
    return AsyncVectorEnv(scenario, num_envs, seed=seed, **cfg)
