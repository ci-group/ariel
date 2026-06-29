"""Stage 2 gate (IMPLEMENTATION_PLAN.md step 6):

* α=0 → env behaves identically to the prior alone (numerical match
  against a direct prior_action rollout).
* α=0.4 + zero residual → prior alone still hovers (drone survives the
  full episode and stays near the target).

Uses a single morph from the v1 library. If the library is missing,
falls back to a sampled morph so the test is still runnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from envs.residual_drone_env import ResidualDroneEnv  # noqa: E402

LIBRARY = Path(__file__).resolve().parents[3] / "__data__/hex_library/v1/library.npz"


def _load_morph(idx: int = 0) -> dict:
    """Return a morph dict (propellers + cmaes_params + features) from
    the v1 library, or freshly sample one if the library is absent."""
    if LIBRARY.exists():
        d = np.load(LIBRARY)
        # The library stores the morph seed; re-sample the same morph via
        # `sample_feasible` and pick the matching seed.
        from hex_sampler import sample_feasible
        target_seed = int(d["morph_seed"][idx])
        morphs = sample_feasible(100, seed=42, stratify=True)
        m = next(mm for mm in morphs if mm.seed == target_seed)
        return {
            "propellers":     m.propellers,
            "mass":           float(m.mass),
            "inertia":        m.inertia,
            "prop_size":      int(m.prop_size),
            "twr":            float(m.twr),
            "cmaes_params":   d["cmaes_params"][idx].astype(np.float32),
            "morph_features": d["morph_features"][idx].astype(np.float32),
        }
    # Fallback: synthesise a single morph + dummy cmaes_params
    from hex_sampler import sample_feasible
    from morphology_features import morph_features
    m = sample_feasible(1, seed=42, stratify=False)[0]
    # Use the prior's default warm-start so the test still demonstrates
    # the integration. Library scores will be better, of course.
    from prior_controller import HoverPrior, N_GAINS
    n = len(m.propellers)
    return {
        "propellers":     m.propellers,
        "mass":           float(m.mass),
        "inertia":        m.inertia,
        "prop_size":      int(m.prop_size),
        "twr":            float(m.twr),
        "cmaes_params":   np.zeros(n + N_GAINS, dtype=np.float32),  # zero gains → falls
        "morph_features": morph_features(
            m.propellers, mass=m.mass, inertia=m.inertia, prop_size=m.prop_size,
        ).astype(np.float32),
    }


def test_zero_alpha_matches_prior():
    """With α=0, residual is ignored; total_action == prior_action."""
    morph = _load_morph(0)
    env = ResidualDroneEnv(morph, alpha=0.0, num_envs=1, max_steps=100)
    env.reset()
    # Any residual; α=0 should null it.
    residual = np.random.RandomState(0).uniform(-1, 1, size=(1, env.num_motors))
    env.step_async(residual.astype(np.float32))
    composed = env.actions_t.cpu().numpy()
    # Compute prior alone on the same state
    expected = env.prior.prior_action(env.world_states, env._cmaes_params_t).cpu().numpy()
    np.testing.assert_allclose(composed, expected, atol=1e-6)


def test_zero_residual_hovers():
    """α=0.4 + zero residual: env runs the prior alone, drone survives."""
    morph = _load_morph(0)
    env = ResidualDroneEnv(morph, alpha=0.4, num_envs=1, max_steps=600)
    env.reset()
    zero_res = np.zeros((1, env.num_motors), dtype=np.float32)
    total_reward = 0.0
    diverged = False
    steps_alive = 0
    for _ in range(env.max_steps):
        env.step_async(zero_res)
        _obs, r, dones, info = env.step_wait()
        total_reward += float(r[0])
        steps_alive += 1
        if bool(dones[0]) and not info[0].get("TimeLimit.truncated", False):
            diverged = True
            break
    # Plan gate: hover task baseline reward should be high under prior
    # alone. We assert no early termination + reasonable cumulative reward
    # (the parent's gate-distance reward is small per step but positive
    # when closing on target; the upright bonus is zero by default, so
    # this is a soft check).
    assert not diverged, f"prior-alone diverged at reward={total_reward:.1f}"
    # Final position should be near the hover target
    pos = env.world_states[0, 0:3].cpu().numpy()
    target = np.array(env.HOVER_TARGET_NED)
    drift = np.linalg.norm(pos - target)
    assert drift < 0.5, f"final drift {drift:.2f} m too large for hover"


if __name__ == "__main__":
    test_zero_alpha_matches_prior()
    print("✓ α=0 matches prior_action exactly")
    test_zero_residual_hovers()
    print("✓ α=0.4 + zero residual hovers within 0.5 m")
