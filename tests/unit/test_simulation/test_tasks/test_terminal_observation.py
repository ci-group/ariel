"""Terminal-observation semantics for drone gate environments.

SB3's `OnPolicyAlgorithm.collect_rollouts` reads `info["terminal_observation"]`
to bootstrap the value function on truncated episodes. If the env returns the
post-reset obs there, the critic is trained to predict `V(reset_distribution)`
instead of the continuation of the terminated episode — a silent value-estimation
bug that breaks long-horizon training.

These tests pin the correct behaviour: terminal_observation must be the state
the drone actually reached before termination, not the subsequent reset state.
"""
from __future__ import annotations

import numpy as np

from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv


def _step_until_done(env, num_envs: int, max_iters: int = 50):
    """Step env with full throttle until at least one env terminates.

    Returns (terminal_obs_at_done, post_reset_obs_at_done, step_count).
    Raises AssertionError if no env terminates within max_iters.
    """
    num_motors = env.num_motors
    for step in range(max_iters):
        act = np.full((num_envs, num_motors), 1.0, dtype=np.float32)
        obs, _rew, done, info = env.step(act)
        if np.any(done):
            idx = int(np.argmax(done))
            term = info[idx]["terminal_observation"]
            return np.asarray(term), np.asarray(obs[idx]), step
    raise AssertionError(f"no env terminated within {max_iters} steps")


def test_torch_terminal_obs_differs_from_reset_obs():
    """Torch env: terminal_observation must not equal the post-reset obs."""
    env = TorchDroneGateEnv(
        num_envs=4, device="cpu", seed=0,
        initialize_at_random_gates=False, max_steps=5,
    )
    env.reset()
    terminal, post_reset, _ = _step_until_done(env, num_envs=4)
    assert not np.allclose(terminal, post_reset), (
        "terminal_observation matched post-reset obs — SB3 bootstrap is broken"
    )


def test_numpy_terminal_obs_differs_from_reset_obs():
    """NumPy env: same contract as Torch env."""
    env = DroneGateEnv(
        num_envs=4, seed=0,
        initialize_at_random_gates=False, max_steps=5,
    )
    env.reset()
    terminal, post_reset, _ = _step_until_done(env, num_envs=4)
    assert not np.allclose(terminal, post_reset), (
        "terminal_observation matched post-reset obs — SB3 bootstrap is broken"
    )


def test_torch_terminal_obs_reflects_last_pre_reset_state():
    """Terminal obs's gate-frame x position should match the drone's actual
    last pre-reset position within one dt of free fall, not snap back to
    the fixed reset offset (-1, 0, 0)."""
    env = TorchDroneGateEnv(
        num_envs=4, device="cpu", seed=0,
        initialize_at_random_gates=False, max_steps=1200,
        x_bounds=(-5, 5), y_bounds=(-5, 5), z_bounds=(-5, 5),
    )
    env.reset()
    # High-throttle rollout → drone exits the z<-5 bound quickly.
    terminal, post_reset, step = _step_until_done(env, num_envs=4, max_iters=500)
    # Post-reset gate-frame z must be ~0 (reset places drone at gate z).
    assert abs(post_reset[2]) < 1e-3, (
        f"post-reset gate-frame z should be ~0, got {post_reset[2]}"
    )
    # Terminal gate-frame z should be well outside the reset neighbourhood —
    # at the very least, it should differ from post_reset by more than one
    # Euler step of free fall (~gravity · dt² ≈ 1e-3).
    assert abs(terminal[2] - post_reset[2]) > 1e-2, (
        f"terminal gate-frame z ({terminal[2]}) and post-reset ({post_reset[2]})"
        f" are suspiciously close; terminal_observation may still be post-reset"
    )


def test_torch_terminal_obs_not_all_zero_or_identical():
    """When multiple envs terminate simultaneously, their terminal_obs values
    should vary across envs — if they all matched some shared post-reset
    constant, SB3 would bootstrap all envs from the same wrong value."""
    env = TorchDroneGateEnv(
        num_envs=8, device="cpu", seed=0,
        initialize_at_random_gates=True, max_steps=20,
    )
    env.reset()
    # Enough steps to force max_steps_reached across the batch.
    for _ in range(25):
        act = np.zeros((8, env.num_motors), dtype=np.float32)
        obs, _rew, done, info = env.step(act)
        if np.sum(done) >= 2:
            terms = [np.asarray(info[i]["terminal_observation"]) for i in range(8) if done[i]]
            # At least two terminals exist — assert they aren't bitwise identical.
            assert not all(np.allclose(terms[0], t) for t in terms[1:]), (
                "all terminal_observations identical across envs — "
                "likely still returning the shared post-reset obs"
            )
            return
    raise AssertionError("expected >= 2 envs to terminate within 25 steps")
