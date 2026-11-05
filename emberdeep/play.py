"""Human play and spectating: ``python -m emberdeep.play --scenario goblin_ambush``.

    python -m emberdeep.play --scenario goblin_ambush --control party_0 --others scripted --record game.edr
    python -m emberdeep.play --scenario cursed_coffer --control party_0,party_1        # hot-seat: several humans
    python -m emberdeep.play --scenario dragon_hoard --watch                           # everybody scripted

Controlled agents act through the renderer's action bar (``renderer.request_human_action``), every other agent
is driven by ``--others`` (``scripted`` hero bots or ``random``).  Scripted monsters always run inside the
environment.  Close the window to quit; after the episode ends the final scoreboard stays on screen.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any


def _parse_size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError as e:
        raise argparse.ArgumentTypeError("size must look like 1600x900") from e


def play(scenario: Any = "goblin_ambush", control: list[str] | None = None, others: str = "scripted",
         seed: int | None = None, record: str | None = None, watch: bool = False, pov: str | None = None,
         size: tuple[int, int] = (1600, 900), quality: str = "medium", sound: bool = True, speed: float = 1.0,
         reaction_mode: str = "agent", episodes: int = 1, hold: bool = True, **cfg: Any) -> dict:
    """Run interactive episodes; returns the metrics of the last one."""
    import emberdeep
    from emberdeep.env.policies import make_controllers
    from emberdeep.replay import Recorder

    probe = emberdeep.make_aec(scenario, **cfg)
    agents = list(probe.possible_agents)
    probe.close()
    if watch:
        control = []
    elif not control:
        control = agents[:1]
    elif control == ["all"]:
        control = agents
    bad = [a for a in control if a not in agents]
    if bad:
        raise SystemExit(f"unknown agent(s) {bad}; scenario agents: {agents}")
    if pov is None:
        pov = control[0] if len(control) == 1 else "omniscient"
    env = emberdeep.make_aec(scenario, render_mode="human", render_width=size[0], render_height=size[1],
                             render_quality=quality, render_sound=sound, render_speed=speed, render_pov=pov,
                             reaction_mode=reaction_mode, **cfg)
    rec = Recorder(env, record) if record else None
    top = rec if rec is not None else env
    core = env.core
    last: dict = {}
    try:
        for ep in range(episodes):
            top.reset(seed=None if seed is None else seed + ep)
            ctl = make_controllers(core, others, [a for a in agents if a not in control], core.last_seed or 0)
            bridge = env._bridge()
            while env.agents and bridge.is_open():
                agent = env.agent_selection
                obs, r, term, trunc, info = env.last()
                if term or trunc:
                    top.step(None)
                    continue
                if agent in control:
                    action = bridge.request_human_action(agent)
                    if not bridge.is_open():
                        break
                    top.step(action)
                else:
                    a, act = ctl[agent].decide()
                    top.step(act if act is not None else a)
                    if watch or agent not in control:
                        env.render()
                        if not bridge.is_open():
                            break
            last = env.episode_metrics()
            if bridge.is_open():
                env.render()
                if hold:
                    bridge.wait_until_closed()
            if not bridge.is_open():
                break
    finally:
        top.close()
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m emberdeep.play", description="Play Emberdeep in the renderer.")
    ap.add_argument("--scenario", default="goblin_ambush")
    ap.add_argument("--control", default=None, help="comma separated agent ids you control, or 'all' "
                                                      "(default: the first agent)")
    ap.add_argument("--others", default="scripted", choices=["scripted", "random"], help="driver of uncontrolled agents")
    ap.add_argument("--watch", action="store_true", help="spectate: every agent is driven by --others")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--record", metavar="FILE.edr", default=None, help="write a replay of the episode")
    ap.add_argument("--pov", default=None, help="omniscient | agent id | faction:<n> (default: your agent)")
    ap.add_argument("--size", type=_parse_size, default=(1600, 900))
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--reaction-mode", default="agent", choices=["auto", "agent"],
                    help="agent: you answer reactions (opportunity attacks, Shield...) yourself")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--no-hold", action="store_true", help="exit right after the episode instead of keeping the "
                                                             "scoreboard on screen")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    control = a.control.split(",") if a.control else None
    m = play(a.scenario, control, a.others, a.seed, a.record, a.watch, a.pov, a.size, a.quality, not a.no_sound,
             a.speed, a.reaction_mode, a.episodes, not a.no_hold)
    g = m.get("global", {})
    print("episode metrics:", {k: round(v, 3) for k, v in g.items() if not k.startswith("team_return/")},
          file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
