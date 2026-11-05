"""Reference MAPPO / IPPO trainer for Emberdeep (PyTorch, CPU).

    # cooperative baseline: 2 heroes vs 3 goblins (about 8 minutes on a laptop CPU)
    python examples/train_mappo.py --scenario goblin_ambush_small --algo mappo --envs 8 --time-limit 8

    # IPPO (decentralised critic), prosocial mixing 0.5
    python examples/train_mappo.py --scenario goblin_ambush --algo ippo --prosocial 0.5

    # two learning factions with one policy each (self-play)
    python examples/train_mappo.py --scenario duel --per-faction --envs 8

Algorithm
    * PPO (clipped surrogate, GAE(lambda) per *decision step*, value clipping + running value normalisation,
      entropy bonus, advantage normalisation, global grad clipping, Adam, linear LR decay).
    * ``--algo mappo``: centralised critic on ``env.state()`` + agent one-hot; ``--algo ippo``: critic on the agent's
      own observation (shared trunk).  The actor is always decentralised.
    * One network shared by all agents (the observation carries class / faction / agent index), or one network per
      faction with ``--per-faction`` (multi-team scenarios).
    * Turn-based credit assignment: only an agent's own decisions are stored; rewards that arrive while it waits are
      summed into its previous decision (see ``examples/rollout.py``).

Outputs in ``--out-dir/<run name>/``: ``metrics.csv`` (one row per update), ``config.json``, ``ckpt_<update>.pt``,
``latest.pt`` and ``final.pt``; TensorBoard event files too if ``tensorboard`` is installed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.dirname(_HERE))            # repo root, so that ``import emberdeep`` works without installing

from models import (ModelSpec, PolicySet, count_params, obs_to_torch, save_checkpoint)    # noqa: E402
from rollout import RolloutCollector                                                       # noqa: E402


# ---------------------------------------------------------------------------------------------- CLI
DENSE_WEIGHTS = {"damage_dealt": 0.05, "kills": 0.2, "damage_taken": -0.002, "death": -0.3}


def parse_kv(items: list[str] | None) -> dict[str, Any]:
    """``["max_steps=200", "enemy_hp='exact'"]`` -> ``{"max_steps": 200, "enemy_hp": "exact"}``."""
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--env-kw expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("environment / social-dilemma knobs")
    g.add_argument("--scenario", default="goblin_ambush_small")
    g.add_argument("--prosocial", type=float, default=None, help="alpha in [0,1]: r' = (1-a) r_i + a mean(r_teammates)")
    g.add_argument("--svo", type=float, default=None, help="social value orientation angle in degrees")
    g.add_argument("--team-reward", action="store_true", default=None, help="every agent gets its team's mean reward")
    g.add_argument("--comm", type=int, default=None, help="SAY vocabulary size V (0 disables communication)")
    g.add_argument("--reaction-mode", choices=["auto", "agent"], default=None,
                   help="'agent': reactions (opportunity attack, Shield, ...) are DECIDE actions of the policy")
    g.add_argument("--reward-scale", type=float, default=None)
    g.add_argument("--max-steps", type=int, default=600,
                   help="truncate episodes after this many env steps (agent decisions); guards against free-action "
                        "stalling (the scenario default is 5000).  Raise it for large scenarios")
    g.add_argument("--reward-preset", choices=["default", "dense"], default="default",
                   help="'dense': stronger damage / kill shaping (damage_dealt 0.05, kills 0.2, damage_taken -0.002, "
                        "death -0.3) for faster learning of combat; changes the payoffs, so not for dilemma studies")
    g.add_argument("--env-kw", action="append", metavar="KEY=VALUE", help="any EnvConfig field, repeatable")
    a = p.add_argument_group("algorithm")
    a.add_argument("--algo", choices=["mappo", "ippo"], default="mappo")
    a.add_argument("--per-faction", action="store_true", help="one policy per faction instead of one shared policy")
    a.add_argument("--gamma", type=float, default=0.99, help="discount per decision step")
    a.add_argument("--gae-lambda", type=float, default=0.95)
    a.add_argument("--clip", type=float, default=0.2)
    a.add_argument("--vf-clip", type=float, default=0.2, help="value clipping range (normalised units); <=0 disables")
    a.add_argument("--vf-coef", type=float, default=0.5)
    a.add_argument("--ent-coef", type=float, default=0.01)
    a.add_argument("--ent-coef-final", type=float, default=None, help="linear anneal target (default: constant)")
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--max-grad-norm", type=float, default=0.5)
    a.add_argument("--epochs", type=int, default=4)
    a.add_argument("--minibatch", type=int, default=128)
    a.add_argument("--target-kl", type=float, default=None, help="stop the epochs early when approx KL exceeds this")
    a.add_argument("--turn-limit", type=int, default=0,
                   help="max regular decisions per turn before only END_TURN stays legal (0 = off, default: the engine now caps turns itself via RulesConfig.max_actions_per_turn and allows each FREE ability once per turn); stops 'stall "
                        "forever with a free action' (see docs/TRAINING.md)")
    a.add_argument("--timeout-penalty", type=float, default=1.0,
                   help="penalty (and no bootstrap) when an episode is truncated: a timeout counts as a defeat. "
                        "Prevents the 'stall forever with a free action' optimum; 0 = bootstrap instead")
    a.add_argument("--step-penalty", type=float, default=0.0, help="extra cost per decision (time pressure)")
    a.add_argument("--no-value-norm", action="store_true")
    a.add_argument("--no-lr-decay", action="store_true")
    m = p.add_argument_group("network")
    m.add_argument("--d-model", type=int, default=64)
    m.add_argument("--hidden", type=int, default=256)
    m.add_argument("--layers", type=int, default=2, help="transformer layers of the entity encoder")
    m.add_argument("--heads", type=int, default=4)
    m.add_argument("--critic-hidden", type=int, default=128)
    r = p.add_argument_group("run")
    r.add_argument("--envs", type=int, default=8, help="parallel environments (worker processes)")
    r.add_argument("--sync", action="store_true", help="run the environments in-process (debugging; implied by --envs 1)")
    r.add_argument("--rollout-steps", type=int, default=64,
                   help="decisions collected per environment per update (batch = envs x rollout-steps)")
    r.add_argument("--total-steps", type=int, default=2_000_000, help="stop after this many env steps")
    r.add_argument("--time-limit", type=float, default=None, help="stop after this many minutes (LR decays accordingly)")
    r.add_argument("--max-updates", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--threads", type=int, default=2, help="torch threads")
    r.add_argument("--out-dir", default="runs")
    r.add_argument("--name", default=None, help="run name (default: <scenario>_<algo>_<timestamp>)")
    r.add_argument("--save-every", type=int, default=50, help="checkpoint every N updates")
    r.add_argument("--log-every", type=int, default=1, help="console line every N updates")
    r.add_argument("--no-tensorboard", action="store_true")
    return p


def env_kwargs_from_args(args) -> dict[str, Any]:
    """EnvConfig keyword arguments for training (memory-friendly dtypes, stale observations for waiting agents)."""
    kw: dict[str, Any] = dict(waiting_obs="stale", obs_dtype="float16", map_dtype="uint8",
                              include_state_in_obs=(args.algo == "mappo"))
    for name, val in (("prosocial", args.prosocial), ("svo", args.svo), ("team_reward", args.team_reward),
                      ("comm_vocab", args.comm), ("reaction_mode", args.reaction_mode),
                      ("reward_scale", args.reward_scale), ("max_steps", args.max_steps)):
        if val is not None:
            kw[name] = val
    if args.reward_preset == "dense":
        kw["reward_weights"] = dict(DENSE_WEIGHTS)
    kw.update(parse_kv(args.env_kw))
    return kw


# ---------------------------------------------------------------------------------------------- logging
class RunLogger:
    """CSV (+ TensorBoard when available).  New columns are added on the fly (the file is rewritten)."""

    def __init__(self, run_dir: str, tensorboard: bool = True):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.rows: list[dict] = []
        self.fields: list[str] = []
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(run_dir)
            except Exception:                                                  # noqa: BLE001 (tensorboard not installed)
                self.tb = None

    def log(self, row: dict, step: int) -> None:
        new = [k for k in row if k not in self.fields]
        self.rows.append(row)
        if new:
            self.fields += new
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, self.fields)
                w.writeheader()
                w.writerows(self.rows)
        else:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, self.fields).writerow(row)
        if self.tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


# ---------------------------------------------------------------------------------------------- PPO
def ppo_update(net, norm, opt, batch: dict[str, np.ndarray], args, lr: float, ent_coef: float,
               rng: np.random.Generator) -> dict[str, float]:
    """Several epochs of clipped-PPO minibatch updates on one policy's batch; returns averaged diagnostics."""
    N = len(batch["adv"])
    obs_keys = [k for k in batch if k not in ("action", "logp", "value", "adv", "ret", "reward")]
    obs = obs_to_torch({k: batch[k] for k in obs_keys if k != "agent_idx"}, batch["agent_idx"])
    act = torch.from_numpy(batch["action"])
    old_logp = torch.from_numpy(batch["logp"])
    adv = torch.from_numpy(batch["adv"])
    if N > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.from_numpy(batch["ret"])
    norm.update(ret)
    tgt = norm.normalize(ret)
    old_vn = norm.normalize(torch.from_numpy(batch["value"]))
    for g in opt.param_groups:
        g["lr"] = lr
    n_mb = max(1, int(round(N / args.minibatch)))
    stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0}
    n_updates = 0
    pred0 = torch.zeros(N)
    stop = False
    for epoch in range(args.epochs):
        perm = rng.permutation(N)
        for chunk in np.array_split(perm, n_mb):
            idx = torch.from_numpy(chunk)
            sub = {k: v[idx] for k, v in obs.items()}
            logp, ent, v = net.evaluate(sub, act[idx])
            logratio = logp - old_logp[idx]
            ratio = logratio.exp()
            a = adv[idx]
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
            if args.vf_clip > 0:
                vc = old_vn[idx] + (v - old_vn[idx]).clamp(-args.vf_clip, args.vf_clip)
                vloss = 0.5 * torch.max((v - tgt[idx]) ** 2, (vc - tgt[idx]) ** 2).mean()
            else:
                vloss = 0.5 * ((v - tgt[idx]) ** 2).mean()
            loss = pg + args.vf_coef * vloss - ent_coef * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - logratio).mean().item()
                stats["pg_loss"] += pg.item()
                stats["v_loss"] += vloss.item()
                stats["entropy"] += ent.mean().item()
                stats["kl"] += kl
                stats["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                stats["grad_norm"] += float(gn)
                if epoch == 0:
                    pred0[idx] = v.detach()
            n_updates += 1
            if args.target_kl is not None and kl > args.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: v / max(n_updates, 1) for k, v in stats.items()}
    var = tgt.var()
    out["explained_var"] = float(1 - (tgt - pred0).var() / (var + 1e-8)) if N > 1 else 0.0
    out["adv_mean_raw"] = float(batch["adv"].mean())
    out["n"] = N
    return out


# ---------------------------------------------------------------------------------------------- setup helpers
def probe_env(scenario: str, env_kw: dict):
    """A throw-away core in the parent process: agents, factions, spaces."""
    from emberdeep.env.core import EmberdeepCore
    return EmberdeepCore(scenario=scenario, **env_kw)


def agent_factions(core) -> list[int]:
    return [int(core.world.entities[core.agent_eid[a]].faction) for a in core.agents]


def build_policies(core, args, env_kw) -> PolicySet:
    factions = agent_factions(core)
    if args.per_faction:
        uniq = sorted(set(factions))
        agent_policy = [uniq.index(f) for f in factions]
        names = [f"faction{f}" for f in uniq]
    else:
        agent_policy, names = [0] * core.n_agents, ["shared"]
    spec = ModelSpec.from_env(core, with_state=bool(env_kw.get("include_state_in_obs")))
    net_cfg = dict(critic="central" if args.algo == "mappo" else "local", d_model=args.d_model, hidden=args.hidden,
                   n_layers=args.layers, n_heads=args.heads, critic_hidden=args.critic_hidden)
    return PolicySet(spec, agent_policy, net_cfg, value_norm=not args.no_value_norm, policy_names=names)


def make_vector(args, env_kw):
    import emberdeep
    sync = args.sync or args.envs == 1
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for k in saved:                                   # workers inherit this: one thread per environment process
        os.environ[k] = "1"
    try:
        return emberdeep.make_vector(args.scenario, num_envs=args.envs, api="parallel", sync=sync, seed=args.seed,
                                     **env_kw)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def summarize_episodes(eps: list[dict], agents: list[str], agent_keys: set[str]) -> dict[str, float]:
    """Mean over finished episodes of: per-agent return / win, and all global + scenario metrics."""
    row: dict[str, float] = {"episodes": float(len(eps))}
    if not eps:
        return row
    for a in agents:
        for k in ("return", "return_raw", "win", "survived"):
            row[f"{k}/{a}"] = float(np.mean([e[a][k] for e in eps]))
    first = eps[0][agents[0]]
    for k in first:
        if k in agent_keys:
            continue
        vals = [e[agents[0]].get(k) for e in eps if isinstance(e[agents[0]].get(k), (int, float))]
        if vals:
            row[f"ep/{k}"] = float(np.mean(vals))
    row["win_rate"] = row.get("ep/win_rate", float("nan"))
    return row


# ---------------------------------------------------------------------------------------------- training
def train(args) -> dict[str, Any]:
    """Run the training loop; returns a summary (final metrics, paths)."""
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env_kw = env_kwargs_from_args(args)
    core = probe_env(args.scenario, env_kw)
    agents = list(core.agents)
    policies = build_policies(core, args, env_kw)
    agent_keys = set(core.episode_metrics()["agents"][agents[0]].keys())
    run_name = args.name or f"{args.scenario}_{args.algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = {"args": vars(args), "env_kwargs": env_kw, "agents": agents, "factions": agent_factions(core),
           "policy_names": policies.policy_names, "agent_policy": policies.agent_policy,
           "params": [count_params(n) for n in policies.nets], "spec": policies.spec.to_dict()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    meta_base = {"scenario": args.scenario, "env_kwargs": env_kw, "agents": agents, "algo": args.algo,
                 "args": vars(args)}
    opts = [torch.optim.Adam(n.parameters(), lr=args.lr, eps=1e-5) for n in policies.nets]
    logger = RunLogger(run_dir, tensorboard=not args.no_tensorboard)
    venv = make_vector(args, env_kw)
    collector = RolloutCollector(policies, args.envs, args.gamma, args.gae_lambda, args.timeout_penalty,
                                  args.step_penalty, args.turn_limit)
    print(f"[train] {args.scenario} algo={args.algo} agents={agents} policies={policies.policy_names} "
          f"params={cfg['params']} run_dir={run_dir}", flush=True)
    obs, infos = venv.reset(seed=args.seed)
    target = args.envs * args.rollout_steps
    t_start = time.time()
    update = 0
    n_resets = 0
    summary: dict[str, Any] = {}
    last_ck = None
    try:
        while True:
            progress = max(collector.env_steps / args.total_steps,
                           (time.time() - t_start) / (60 * args.time_limit) if args.time_limit else 0.0)
            if progress >= 1.0 or (args.max_updates is not None and update >= args.max_updates):
                break
            lr = args.lr if args.no_lr_decay else args.lr * max(0.02, 1.0 - progress)
            ef = args.ent_coef if args.ent_coef_final is None else \
                args.ent_coef + (args.ent_coef_final - args.ent_coef) * min(progress, 1.0)
            t0 = time.time()
            steps0 = collector.env_steps
            dec0 = collector.decisions
            # ---- rollout
            while collector.n_ready < target:
                if env_kw.get("skip_scripted", True) and not all(any(i["acting"]) for i in infos):
                    # an environment whose fresh episode is already over (rare: the scripted opening decided the
                    # game; docs/dev/contract_requests/train.md) would crash on step(): restart every environment
                    n_resets += 1
                    obs, infos = venv.reset(seed=args.seed + 100_003 * n_resets)
                    collector.open.clear()
                    collector.limiter.reset()
                    continue
                actions = collector.act(obs, infos)
                obs, rewards, terms, truncs, infos = venv.step(actions)
                collector.observe(rewards, terms, truncs, infos)
            t_roll = time.time() - t0
            batches = collector.drain()
            # ---- optimisation
            t1 = time.time()
            row: dict[str, float] = {}
            for p, batch in batches.items():
                st = ppo_update(policies.nets[p], policies.norms[p], opts[p], batch, args, lr, ef, rng)
                tag = "" if len(policies) == 1 else f"/{policies.policy_names[p]}"
                for k, v in st.items():
                    row[f"{k}{tag}"] = v
                row[f"reward_per_decision{tag}"] = float(batch["reward"].mean())
            update += 1
            t_opt = time.time() - t1
            dt = time.time() - t0
            eps = collector.pop_episodes()
            row.update(summarize_episodes(eps, agents, agent_keys))
            row.update({"update": update, "env_steps": collector.env_steps, "decisions": collector.decisions,
                        "timeouts": collector.n_timeouts, "sps": (collector.env_steps - steps0) / dt, "t_rollout": t_roll, "t_optim": t_opt,
                        "time": time.time() - t_start, "lr": lr, "ent_coef": ef})
            logger.log(row, collector.env_steps)
            summary = row
            if update % args.log_every == 0:
                wr = row.get("win_rate", float("nan"))
                if len(set(cfg["factions"])) > 1:                       # competitive: show each agent's win rate
                    wr_txt = "/".join(f"{row.get(f'win/{a}', float('nan')):.2f}" for a in agents)
                else:
                    wr_txt = f"{wr:5.2f}"
                rets = " ".join(f"{a}:{row[f'return/{a}']:+.2f}" for a in agents if f"return/{a}" in row)
                sfx = "" if len(policies) == 1 else f"/{policies.policy_names[0]}"
                print(f"[upd {update:4d}] steps={collector.env_steps:8d} sps={row['sps']:6.0f} "
                      f"win={wr_txt} ep={int(row['episodes']):3d} ret[{rets}] ent={row.get('entropy' + sfx, float('nan')):.2f} "
                      f"kl={row.get('kl' + sfx, float('nan')):.4f} ev={row.get('explained_var' + sfx, float('nan')):+.2f} "
                      f"t={row['time']:.0f}s", flush=True)
            if update % args.save_every == 0:
                last_ck = os.path.join(run_dir, f"ckpt_{update}.pt")
                meta = {**meta_base, "update": update, "env_steps": collector.env_steps}
                save_checkpoint(last_ck, policies, meta)
                save_checkpoint(os.path.join(run_dir, "latest.pt"), policies, meta)
    finally:
        venv.close()
        logger.close()
    final = os.path.join(run_dir, "final.pt")
    save_checkpoint(final, policies, {**meta_base, "update": update, "env_steps": collector.env_steps})
    save_checkpoint(os.path.join(run_dir, "latest.pt"), policies,
                    {**meta_base, "update": update, "env_steps": collector.env_steps})
    print(f"[train] done: {update} updates, {collector.env_steps} env steps, {time.time() - t_start:.0f}s -> {final}",
          flush=True)
    return {"run_dir": run_dir, "final": final, "updates": update, "env_steps": collector.env_steps, "last_row": summary,
            "metrics_csv": logger.path}


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
