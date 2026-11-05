"""(Golden values re-baselined after the once-per-turn FREE ability rule, and again for v1.1.)
v1.1 re-baseline (rules-engine): digests changed because of intentional rules semantics - low walls are climbable obstacles,
carrying-capacity speed limits (coins weigh 1 lb / 50), attunement-gated items, stable recovery, underwater / projectile
rules, mounts, new RulesConfig fields, extra RNG draws; status block grew 18 -> 22 bits (MOUNTED, CARRYING_RIDER, SQUEEZING, BURROWED).  Observation/action layout is unchanged.
Bit-identity regression: observation / mask / reward digests for fixed seeds and action sequences.

These golden values pin the observation and action semantics; performance work must not change them.
Regenerate (only after an intentional semantic change) with ``python tests/test_env_obs_regression.py``."""
import hashlib

import numpy as np
import pytest

import emberdeep
from helpers_env import clean_defs  # noqa: F401 (autouse fixture)

CASES = [("goblin_ambush", 3, 90), ("dragon_hoard", 5, 60), ("skirmish_2v2", 1, 60), ("cursed_coffer", 2, 60)]
GOLDEN = {
    ('goblin_ambush', 3, 90): '6fc9d1ab81d63aba',
    ('dragon_hoard', 5, 60): '2ee77937f1237480',
    ('skirmish_2v2', 1, 60): 'e16024d894e6740b',
    ('cursed_coffer', 2, 60): '61e7e93451e2a94a',
}


def digest_run(scn, seed, n):
    env = emberdeep.make_aec(scn)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    h = hashlib.sha1()
    steps = 0
    while env.agents and steps < n:
        agent = env.agent_selection
        obs, r, te, tr, _ = env.last()
        h.update(agent.encode())
        for k in sorted(obs):
            h.update(np.ascontiguousarray(obs[k]).tobytes())
        h.update(repr(round(r, 12)).encode())
        if te or tr:
            env.step(None)
            continue
        env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        steps += 1
    h.update(repr(env.core.world.state_hash()).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("scn,seed,n", CASES)
def test_observation_digest_is_stable(scn, seed, n):
    assert digest_run(scn, seed, n) == GOLDEN[(scn, seed, n)]


if __name__ == "__main__":
    for c in CASES:
        print(f"    {c!r}: {digest_run(*c)!r},")
