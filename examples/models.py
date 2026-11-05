"""Neural networks for the Emberdeep reference trainers (PyTorch, CPU friendly).

The policy exploits the *structure* of the Dict observation and of the flat action space (docs/ENV_API.md)::

    map [C,21,21] --CNN--> 128 ----------------------------------+
    self [F_self] --MLP--> 64 ----------+                         |
    entities [E,F]+entity_ids --MLP+emb-+--> Transformer (slot 0  +--> trunk MLP --> h [256]
        (entity_mask = key padding)         = self / CLS token)    |        |
    abilities [K,F]+ability_ids --MLP+emb--> A_k [K,64] --mean-----+        |
    messages [n_agents,V] --Linear--> 32 -------------------------+         |
    agent index --Embedding--> 16 ----------------------------------+        |
                                                                              v
    heads:  END/MOVE/ASCEND/DESCEND  <- Linear(h)                       [11]
            ABILITY k on target t    <- <q_k, W T_t>/sqrt(d) + b_k      [K*T]   T_t = entity token (t<E) or
            APPROACH e               <- MLP([T_e, h])                   [E]            learned point embedding
            SAY v / DECIDE d         <- Linear(h)                       [V] [D]
         concatenated in the flat layout of ``ActionLayout`` and masked with ``action_mask`` (-1e9)
    value:  IPPO  -> MLP(h)                      (decentralised critic on the agent's own observation)
            MAPPO -> MLP([env.state(), agent id])  (centralised critic)

Everything shape-related is derived from a :class:`ModelSpec` (built from the environment), so the same classes
work for any ``EnvConfig`` (``max_entities``, ``n_point_targets``, ``comm_vocab``, ``view_radius`` ...).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BASIC = 11                     # END_TURN, 8 x MOVE, ASCEND, DESCEND
MASK_VALUE = -1e9


# ---------------------------------------------------------------------------------------------- spec
@dataclass
class ModelSpec:
    """Dimensions of the observation / action space a network is built for."""
    f_self: int
    f_ent: int
    f_ab: int
    map_channels: int
    map_size: int
    n_entity_ids: int
    n_ability_ids: int
    E: int                        # entity slots
    K: int                        # ability slots
    P: int                        # point-target slots
    V: int                        # SAY tokens (0 = no communication)
    D: int                        # DECIDE options
    n_agents: int
    state_dim: int = 0            # 0: no global state (IPPO)

    @property
    def T(self) -> int:
        return self.E + self.P

    @property
    def A(self) -> int:
        return N_BASIC + self.K * self.T + self.E + self.V + self.D

    @staticmethod
    def from_env(core, with_state: bool) -> "ModelSpec":
        """Build from an ``EmberdeepCore`` (or anything exposing ``observation_space``, ``layout``, ``n_agents``,
        ``state_space``)."""
        sp = core.observation_space.spaces
        lay = core.layout
        return ModelSpec(
            f_self=sp["self"].shape[0], f_ent=sp["entities"].shape[1], f_ab=sp["abilities"].shape[1],
            map_channels=sp["map"].shape[0], map_size=sp["map"].shape[1],
            n_entity_ids=int(sp["entity_ids"].high.max()) + 1, n_ability_ids=int(sp["ability_ids"].high.max()) + 1,
            E=lay.E, K=lay.K, P=lay.P, V=lay.V, D=lay.D, n_agents=core.n_agents,
            state_dim=int(core.state_space.shape[0]) if with_state else 0)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------------------------- obs -> tensors
OBS_KEYS = ("self", "entities", "entity_mask", "entity_ids", "abilities", "ability_ids", "map", "messages",
            "action_mask", "ability_mask")


def obs_to_torch(obs: dict[str, np.ndarray], agent_idx: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    """Batched numpy observation (leading batch dim) -> tensors the network consumes.

    Handles ``obs_dtype=float16`` and ``map_dtype=uint8`` (scaled back to [0, 1])."""
    out: dict[str, torch.Tensor] = {}
    for k in ("self", "entities", "abilities"):
        out[k] = torch.from_numpy(np.ascontiguousarray(obs[k])).float()
    m = np.ascontiguousarray(obs["map"])
    out["map"] = torch.from_numpy(m).float().div_(255.0) if m.dtype == np.uint8 else torch.from_numpy(m).float()
    out["entity_mask"] = torch.from_numpy(np.ascontiguousarray(obs["entity_mask"])).bool()
    out["action_mask"] = torch.from_numpy(np.ascontiguousarray(obs["action_mask"])).bool()
    if "ability_mask" in obs:
        out["ability_mask"] = torch.from_numpy(np.ascontiguousarray(obs["ability_mask"])).bool()
    out["entity_ids"] = torch.from_numpy(np.ascontiguousarray(obs["entity_ids"])).long()
    out["ability_ids"] = torch.from_numpy(np.ascontiguousarray(obs["ability_ids"])).long()
    if "messages" in obs:
        out["messages"] = torch.from_numpy(np.ascontiguousarray(obs["messages"])).float()
    if "state" in obs:
        out["state"] = torch.from_numpy(np.ascontiguousarray(obs["state"])).float()
    if agent_idx is not None:
        out["agent_idx"] = torch.from_numpy(np.asarray(agent_idx, dtype=np.int64))
    return out


# ---------------------------------------------------------------------------------------------- helpers
def _init(m: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
    return m


def mlp(sizes: list[int], last_gain: float | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        last = i == len(sizes) - 2
        _init(lin, last_gain if (last and last_gain is not None) else math.sqrt(2))
        layers.append(lin)
        if not last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def assemble_flat_logits(basic: torch.Tensor, ability: torch.Tensor, approach: torch.Tensor,
                         say: torch.Tensor | None, decide: torch.Tensor) -> torch.Tensor:
    """Concatenate the factored logits in the env's flat layout (``ActionLayout``)::

        [0..10]                 END_TURN, MOVE x8, ASCEND, DESCEND            basic   [B, 11]
        11 + k*T + t            ABILITY k on target slot t                     ability [B, K, T]
        11 + K*T + e            APPROACH entity slot e                         approach [B, E]
        11 + K*T + E + v        SAY token v                                    say [B, V]
        11 + K*T + E + V + d    DECIDE option d                                decide [B, D]
    """
    parts = [basic, ability.reshape(ability.shape[0], -1), approach]
    if say is not None and say.shape[-1] > 0:
        parts.append(say)
    parts.append(decide)
    return torch.cat(parts, dim=-1)


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Illegal actions get -1e9 (a legal action always exists: END_TURN / NOOP / a DECIDE option)."""
    return logits.masked_fill(~action_mask, MASK_VALUE)


# ---------------------------------------------------------------------------------------------- the network
class EmberdeepNet(nn.Module):
    """Actor-critic for one policy.  ``critic="central"`` (MAPPO) needs ``obs["state"]``; ``"local"`` (IPPO) shares
    the actor trunk."""

    def __init__(self, spec: ModelSpec, critic: str = "local", d_model: int = 64, hidden: int = 256,
                 n_layers: int = 2, n_heads: int = 4, critic_hidden: int = 128):
        super().__init__()
        if critic not in ("local", "central"):
            raise ValueError("critic must be 'local' or 'central'")
        if critic == "central" and spec.state_dim <= 0:
            raise ValueError("a central critic needs spec.state_dim > 0 (environment built with include_state_in_obs)")
        self.spec, self.critic_kind = spec, critic
        self.cfg = dict(critic=critic, d_model=d_model, hidden=hidden, n_layers=n_layers, n_heads=n_heads,
                        critic_hidden=critic_hidden)
        s, d = spec, d_model
        self.d = d
        # --- map CNN: C x 21 x 21 -> 128
        self.cnn = nn.Sequential(
            _init(nn.Conv2d(s.map_channels, 32, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(), nn.Flatten())
        side = s.map_size
        for _ in range(2):
            side = (side - 1) // 2 + 1
        self.cnn_out = nn.Sequential(_init(nn.Linear(64 * side * side, 128)), nn.ReLU())
        # --- self features
        self.self_mlp = mlp([s.f_self, d])
        self.self_to_token = _init(nn.Linear(d, d))
        # --- entity encoder (+ transformer; slot 0 = self acts as CLS / query)
        self.ent_in = _init(nn.Linear(s.f_ent, d))
        self.ent_emb = nn.Embedding(s.n_entity_ids, d, padding_idx=0)
        self.ent_norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        # --- abilities
        self.ab_in = _init(nn.Linear(s.f_ab, d))
        self.ab_emb = nn.Embedding(s.n_ability_ids, d, padding_idx=0)
        self.ab_out = mlp([d, d])
        # --- messages / agent id
        self.msg = mlp([s.n_agents * s.V, 32]) if s.V > 0 else None
        self.agent_emb = nn.Embedding(max(s.n_agents, 1), 16)
        # --- trunk
        trunk_in = 128 + d + d + d + (32 if s.V > 0 else 0) + 16
        self.trunk = mlp([trunk_in, hidden, hidden])
        self.trunk_act = nn.ReLU()
        # --- policy heads
        self.basic_head = mlp([hidden, N_BASIC], last_gain=0.01)
        self.dp = d
        self.h_to_q = _init(nn.Linear(hidden, d))
        self.h_to_pt = _init(nn.Linear(hidden, d))
        self.q_proj = _init(nn.Linear(d, d), 1.0)
        self.k_proj = _init(nn.Linear(d, d), 1.0)
        self.ab_bias = nn.Linear(d, 1)
        nn.init.zeros_(self.ab_bias.weight)
        nn.init.zeros_(self.ab_bias.bias)
        self.point_emb = nn.Parameter(torch.randn(max(s.P, 1), d) * 0.1)
        self.h_to_app = _init(nn.Linear(hidden, d))
        self.approach_head = mlp([2 * d, 64, 1], last_gain=0.01)
        self.say_head = mlp([hidden, s.V], last_gain=0.01) if s.V > 0 else None
        self.decide_head = mlp([hidden, s.D], last_gain=0.01)
        # --- critic
        if critic == "local":
            self.value_head = mlp([hidden, critic_hidden, 1], last_gain=1.0)
            self.critic_net = None
        else:
            self.critic_net = mlp([s.state_dim + s.n_agents, critic_hidden * 2, critic_hidden, 1], last_gain=1.0)
            self.value_head = None

    # ------------------------------------------------------------------------------------ encoding
    def encode(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Shared encoders -> trunk ``h [B, hidden]``, entity tokens ``tok [B, E, d]``, ability embeddings."""
        B = obs["self"].shape[0]
        emask = obs["entity_mask"]
        self_f = self.self_mlp(obs["self"])
        # entity tokens
        x = self.ent_in(obs["entities"]) + self.ent_emb(obs["entity_ids"])
        x = x + F.pad(self.self_to_token(self_f)[:, None, :], (0, 0, 0, self.spec.E - 1))   # self state into slot 0
        x = self.ent_norm(x)
        pad = ~emask
        pad = pad.clone()
        pad[:, 0] = False                                     # slot 0 (self) is always attendable
        tok = self.transformer(x, src_key_padding_mask=pad)
        # abilities
        ab = self.ab_out(F.relu(self.ab_in(obs["abilities"]) + self.ab_emb(obs["ability_ids"])))
        am = obs["abilities"].new_ones(B, self.spec.K, 1) if "ability_mask" not in obs else \
            obs["ability_mask"].float().unsqueeze(-1)
        ab_pool = (ab * am).sum(1) / am.sum(1).clamp(min=1.0)
        feats = [self.cnn_out(self.cnn(obs["map"])), tok[:, 0], self_f, ab_pool]
        if self.msg is not None:
            feats.append(self.msg(obs["messages"].reshape(B, -1)))
        aidx = obs.get("agent_idx")
        if aidx is None:
            aidx = torch.zeros(B, dtype=torch.long)
        feats.append(self.agent_emb(aidx))
        h = self.trunk_act(self.trunk(torch.cat(feats, dim=-1)))
        return {"h": h, "tok": tok, "ab": ab}

    # ------------------------------------------------------------------------------------ heads
    def logit_parts(self, enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The factored logits (before assembly / masking)."""
        h, tok, ab = enc["h"], enc["tok"], enc["ab"]
        s = self.spec
        B = h.shape[0]
        basic = self.basic_head(h)
        q = self.q_proj(ab) + self.h_to_q(h)[:, None, :]                                  # [B,K,d]
        keys = tok
        if s.P > 0:
            pts = self.point_emb[None, : s.P] + self.h_to_pt(h)[:, None, :]               # [B,P,d]
            keys = torch.cat([tok, pts], dim=1)                                           # [B,T,d]
        keys = self.k_proj(keys)
        ability = torch.einsum("bkd,btd->bkt", q, keys) / math.sqrt(self.dp) + self.ab_bias(ab)   # [B,K,T]
        app_in = torch.cat([tok, self.h_to_app(h)[:, None, :].expand(B, s.E, self.d)], dim=-1)
        approach = self.approach_head(app_in).squeeze(-1)                                 # [B,E]
        say = self.say_head(h) if self.say_head is not None else None
        decide = self.decide_head(h)
        return {"basic": basic, "ability": ability, "approach": approach, "say": say, "decide": decide}

    def logits(self, obs: dict[str, torch.Tensor], enc: dict | None = None) -> torch.Tensor:
        enc = enc if enc is not None else self.encode(obs)
        p = self.logit_parts(enc)
        flat = assemble_flat_logits(p["basic"], p["ability"], p["approach"], p["say"], p["decide"])
        return mask_logits(flat, obs["action_mask"])

    def value_from(self, obs: dict[str, torch.Tensor], enc: dict | None = None) -> torch.Tensor:
        """Value in the (possibly normalised) scale of the critic head, ``[B]``."""
        if self.critic_kind == "central":
            onehot = F.one_hot(obs["agent_idx"], self.spec.n_agents).float()
            return self.critic_net(torch.cat([obs["state"], onehot], dim=-1)).squeeze(-1)
        enc = enc if enc is not None else self.encode(obs)
        return self.value_head(enc["h"]).squeeze(-1)

    def forward(self, obs: dict[str, torch.Tensor]):
        enc = self.encode(obs)
        return self.logits(obs, enc), self.value_from(obs, enc)

    def evaluate(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        """``(logp, entropy, value)`` of ``actions`` under the current parameters (training)."""
        logits, value = self(obs)
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, actions[:, None]).squeeze(-1)
        p = logp_all.exp()
        ent = -(p * logp_all.clamp(min=-1e4)).sum(-1)
        return logp, ent, value

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        """``(action, logp, value, probs)`` for a batch (rollouts / evaluation)."""
        logits, value = self(obs)
        logp_all = F.log_softmax(logits, dim=-1)
        if deterministic:
            a = logp_all.argmax(-1)
        else:
            a = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
        return a, logp_all.gather(-1, a[:, None]).squeeze(-1), value, logp_all.exp()

    @torch.no_grad()
    def value_only(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_from(obs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------------------------- value norm
class ValueNorm(nn.Module):
    """Running mean / std of the value targets (as in MAPPO's ``ValueNorm``): the critic regresses normalised
    returns; ``denormalize`` gives values in reward units."""

    def __init__(self, window: float = 2e5, eps: float = 1e-4):
        super().__init__()
        self.window, self.eps = window, eps
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = float(x.numel())
        if n == 0:
            return
        bm, bv = x.mean(), x.var(unbiased=False)
        self.count += n
        w = n / min(float(self.count), self.window)
        old = self.mean.clone()
        self.mean += w * (bm - old)
        self.var.copy_((1 - w) * self.var + w * bv + w * (1 - w) * (bm - old) ** 2)

    @property
    def std(self) -> torch.Tensor:
        return self.var.clamp(min=self.eps).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class IdentityNorm(ValueNorm):
    """Drop-in that does nothing (``--no-value-norm``)."""

    def update(self, x: torch.Tensor) -> None:
        pass

    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x


# ---------------------------------------------------------------------------------------------- policy set
class PolicySet:
    """Networks + value normalisers; ``agent_policy[j]`` is the policy index of agent ``j``.

    One shared policy (default) or one per faction (``--per-faction``)."""

    def __init__(self, spec: ModelSpec, agent_policy: list[int], net_cfg: dict, value_norm: bool = True,
                 policy_names: list[str] | None = None):
        self.spec, self.agent_policy, self.net_cfg = spec, list(agent_policy), dict(net_cfg)
        n = max(agent_policy) + 1
        self.policy_names = policy_names or [f"policy_{i}" for i in range(n)]
        self.nets = [EmberdeepNet(spec, **net_cfg) for _ in range(n)]
        self.norms = [(ValueNorm() if value_norm else IdentityNorm()) for _ in range(n)]
        self.value_norm = value_norm

    def __len__(self) -> int:
        return len(self.nets)

    def eval(self) -> "PolicySet":
        for m in (*self.nets, *self.norms):
            m.eval()
        return self

    def state_dict(self) -> dict:
        return {"nets": [n.state_dict() for n in self.nets], "norms": [v.state_dict() for v in self.norms]}

    def load_state_dict(self, sd: dict) -> None:
        for n, s in zip(self.nets, sd["nets"]):
            n.load_state_dict(s)
        for v, s in zip(self.norms, sd["norms"]):
            v.load_state_dict(s)


def save_checkpoint(path: str, policies: PolicySet, meta: dict) -> None:
    """``meta``: JSON-able run information (args, env kwargs, agents, counters)."""
    torch.save({"format": "emberdeep-baseline-v1", "spec": policies.spec.to_dict(), "agent_policy": policies.agent_policy,
                "net_cfg": policies.net_cfg, "value_norm": policies.value_norm, "policy_names": policies.policy_names,
                "weights": policies.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str) -> tuple[PolicySet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "emberdeep-baseline-v1":
        raise ValueError(f"{path} is not an Emberdeep baseline checkpoint")
    ps = PolicySet(ModelSpec(**ck["spec"]), ck["agent_policy"], ck["net_cfg"], ck["value_norm"], ck["policy_names"])
    ps.load_state_dict(ck["weights"])
    return ps.eval(), ck["meta"]


def policy_forward_by_group(policies: PolicySet, obs_np: dict[str, np.ndarray], agent_idx: np.ndarray,
                            deterministic: bool = False):
    """Run each agent's policy on a batch of observations (rows may belong to different agents / policies).

    Returns numpy ``actions, logp, value (denormalised), probs`` aligned with the rows."""
    n = len(agent_idx)
    pol_of_row = np.array([policies.agent_policy[j] for j in agent_idx])
    actions = np.zeros(n, np.int64)
    logp = np.zeros(n, np.float32)
    value = np.zeros(n, np.float32)
    probs = np.zeros((n, policies.spec.A), np.float32)
    for p in np.unique(pol_of_row):
        rows = np.flatnonzero(pol_of_row == p)
        t = obs_to_torch({k: v[rows] for k, v in obs_np.items()}, agent_idx[rows])
        a, lp, v, pr = policies.nets[p].act(t, deterministic)
        actions[rows] = a.numpy()
        logp[rows] = lp.numpy()
        value[rows] = policies.norms[p].denormalize(v).numpy()
        probs[rows] = pr.numpy()
    return actions, logp, value, probs
