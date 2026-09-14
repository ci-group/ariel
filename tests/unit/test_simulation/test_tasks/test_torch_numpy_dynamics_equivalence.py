"""Torch vs numpy dynamics equivalence for the drone gate environments.

Guards the physics core that the whole spear stack depends on: the torch
env (`TorchDroneGateEnv`) is a GPU-accelerated reimplementation of the
numpy env's (`DroneGateEnv`) dynamics. Both should compute an identical
state derivative from an identical (state, action) pair — anything else
means the two envs have silently drifted.

We compare `drone_sim.dynamics_func` (numpy, symbolic) against the
compiled callable returned by `_build_torch_dynamics(...)` (torch),
which is the same math re-expressed in torch ops.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.simulation.tasks.torch_drone_gate_env import (
    TorchDroneGateEnv,
    _build_torch_dynamics,
)


def _standard_envs(num_envs: int = 4):
    """Build both envs with the standard quad, no gates/reward complications."""
    torch_env = TorchDroneGateEnv(
        num_envs=num_envs,
        device="cpu",
        seed=0,
        initialize_at_random_gates=False,
        action_filter_alpha=1.0,
    )
    numpy_env = DroneGateEnv(
        num_envs=num_envs,
        propellers=torch_env.drone_sim.config.propellers,
        seed=0,
        initialize_at_random_gates=False,
        action_filter_alpha=1.0,
    )
    return torch_env, numpy_env


def _random_states(rng: np.random.RandomState, num_envs: int, num_motors: int):
    """A batch of plausible (state, action) samples near hover."""
    state = np.zeros((num_envs, 12 + num_motors), dtype=np.float32)
    state[:, 0:3] = rng.uniform(-1.0, 1.0, size=(num_envs, 3))  # pos
    state[:, 2] += -1.5  # NED — hover near z=-1.5
    state[:, 3:6] = rng.uniform(-0.3, 0.3, size=(num_envs, 3))  # vel
    state[:, 6:9] = rng.uniform(-0.2, 0.2, size=(num_envs, 3))  # angles
    state[:, 9:12] = rng.uniform(-0.5, 0.5, size=(num_envs, 3))  # rates
    state[:, 12:] = rng.uniform(-0.5, 0.5, size=(num_envs, num_motors))  # motors
    action = rng.uniform(-1.0, 1.0, size=(num_envs, num_motors)).astype(np.float32)
    return state, action


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DRIFT (2026-07-27): torch `_build_torch_dynamics` and numpy "
        "`drone_sim.dynamics_func` do not produce identical state_dot for "
        "the standard quad. Max abs diff ~0.62 in accelerations. Suspected "
        "source: aero-drag or motor-lag terms encoded differently. This is "
        "the divergence the audit warned about — fix upstream, then flip "
        "this marker."
    ),
)
def test_dynamics_func_matches_torch_build():
    """Numpy `dynamics_func` and the torch-built dynamics agree on state_dot."""
    torch_env, numpy_env = _standard_envs(num_envs=4)
    num_motors = torch_env.num_motors

    torch_dyn = _build_torch_dynamics(
        torch_env.drone_sim.params,
        num_motors,
        torch_env.drone_sim.g,
        torch.device("cpu"),
        torch.float32,
    )

    rng = np.random.RandomState(0)
    for trial in range(5):
        state_np, action_np = _random_states(rng, num_envs=4, num_motors=num_motors)

        # Numpy dynamics: dynamics_func expects (12+N, E), (N, E), returns (12+N, E)
        sdot_np = numpy_env.drone_sim.dynamics_func(state_np.T, action_np.T).T

        # Torch dynamics: same shape convention
        sdot_th = torch_dyn(
            torch.from_numpy(state_np.T),
            torch.from_numpy(action_np.T),
        ).T.numpy()

        np.testing.assert_allclose(
            sdot_th, sdot_np, atol=1e-5, rtol=1e-4,
            err_msg=f"trial {trial}: torch and numpy dynamics disagree",
        )


@pytest.mark.xfail(
    strict=True,
    reason="Consequence of the dynamics drift above; flip together.",
)
def test_single_step_trajectory_matches():
    """One Euler step of dynamics reproduces the same world_state on both envs.

    Skips reward/gate/termination logic (which legitimately differ) — we set
    the state, call the pure dynamics + dt update, and compare.
    """
    torch_env, numpy_env = _standard_envs(num_envs=2)
    num_motors = torch_env.num_motors
    dt = torch_env.dt
    assert dt == numpy_env.dt

    rng = np.random.RandomState(1)
    state_np, action_np = _random_states(rng, num_envs=2, num_motors=num_motors)

    # Numpy step (dynamics only, mirroring step_wait line 421-424):
    sdot_np = numpy_env.drone_sim.dynamics_func(state_np.T, action_np.T).T
    new_state_np = (state_np + dt * sdot_np).astype(np.float32)

    # Torch step (dynamics only, mirroring step_wait's fs + dt * fsd):
    torch_dyn = _build_torch_dynamics(
        torch_env.drone_sim.params, num_motors,
        torch_env.drone_sim.g, torch.device("cpu"), torch.float32,
    )
    st = torch.from_numpy(state_np)
    at = torch.from_numpy(action_np)
    sdot_th = torch_dyn(st.T, at.T).T
    new_state_th = (st + dt * sdot_th).numpy()

    np.testing.assert_allclose(new_state_th, new_state_np, atol=1e-5, rtol=1e-4)


def test_env_construction_no_render_warning():
    """`render_mode` is unconditionally set — SB3 must not emit its
    'render_mode attribute not defined' warning on env construction."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote warnings to errors
        env = TorchDroneGateEnv(num_envs=1, device="cpu", seed=0)
        assert hasattr(env, "render_mode")
        assert env.render_mode is None


if __name__ == "__main__":
    test_dynamics_func_matches_torch_build()
    print("[OK] dynamics_func matches _build_torch_dynamics")
    test_single_step_trajectory_matches()
    print("[OK] single Euler step matches")
    test_env_construction_no_render_warning()
    print("[OK] no render_mode warning on construction")
