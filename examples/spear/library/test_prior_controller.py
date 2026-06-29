"""Tests for the analytical hover prior.

Most important test: ``test_pitch_disturbance_is_damped`` and
``test_roll_disturbance_is_damped`` — these are the sign-convention
guards that would have caught the original positive-feedback pitch bug.
If these ever fail, **do not weaken them** — find the regression first.

Run:
    uv run pytest examples/spear/library/test_prior_controller.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ariel.simulation.drone.dynamics_params import derive_reference_params
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics

sys.path.insert(0, str(Path(__file__).parent))
from hex_sampler import sample_feasible              # noqa: E402
from prior_controller import (                       # noqa: E402
    HoverPrior,
    N_GAINS,
    compute_u_hover,
    tilt_mixer,
    yaw_mixer,
)

HOVER_TARGET_NED = (0.0, 0.0, -1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hex_setup():
    """One feasible hexacopter morphology plus all the bits the prior
    needs at construction time. Shared across tests because constructing
    DroneConfiguration is the slow part."""
    morphs = sample_feasible(1, seed=2026, stratify=False)
    morph = morphs[0]
    params = derive_reference_params(
        propellers=morph.propellers, mass=morph.mass,
        inertia=morph.inertia, prop_size=morph.prop_size,
    )
    prior = HoverPrior(
        propellers=morph.propellers, params=params,
        target_ned=HOVER_TARGET_NED,
    )
    return morph, params, prior


# ─────────────────────────────────────────────────────────────────────────────
# Mixer signs — pure math, no dynamics
# ─────────────────────────────────────────────────────────────────────────────

def test_tilt_mixer_pitch_column_is_negative_cos():
    """For a motor at (+x, 0, 0) the pitch column must be -1, not +1.
    This is the literal bit the original 35c had backwards."""
    propellers = [{"loc": (0.12, 0.0, 0.0), "dir": ("nan",) * 3 + ("ccw",)}]
    mix = tilt_mixer(propellers)
    assert mix[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert mix[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_tilt_mixer_roll_column_is_sin():
    """For a motor at (0, +y, 0) the roll column must be +1."""
    propellers = [{"loc": (0.0, 0.12, 0.0), "dir": ("nan",) * 3 + ("ccw",)}]
    mix = tilt_mixer(propellers)
    assert mix[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert mix[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_yaw_mixer_signs_negate_spin():
    """yaw_mix[i] must be -spin_sign(i); sign is baked into the mixer
    so the gain stays positive."""
    propellers = [
        {"loc": (0.0, 0.0, 0.0), "dir": ("nan",) * 3 + ("ccw",)},
        {"loc": (0.0, 0.0, 0.0), "dir": ("nan",) * 3 + ("cw",)},
    ]
    ymix = yaw_mixer(propellers)
    assert ymix[0] == pytest.approx(-1.0)   # ccw → -1
    assert ymix[1] == pytest.approx(+1.0)   # cw  → +1


# ─────────────────────────────────────────────────────────────────────────────
# u_hover
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_u_hover_in_valid_range(hex_setup):
    _, params, prior = hex_setup
    u = compute_u_hover(params, prior.n_motors, prior.gravity)
    assert -1.0 < u < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Class construction + parameter helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_prior_param_dim_is_n_plus_5(hex_setup):
    _, _, prior = hex_setup
    assert prior.param_dim == prior.n_motors + N_GAINS == prior.n_motors + 5


def test_default_init_params_matches_35c_warmstart(hex_setup):
    """Without morph args, the warm-start matches the original 35c
    hand-tuned values (back-compat)."""
    _, _, prior = hex_setup
    init = prior.default_init_params()
    N = prior.n_motors
    assert init.shape == (prior.param_dim,)
    np.testing.assert_array_equal(init[:N], np.zeros(N, dtype=np.float32))
    assert init[N + 0] == pytest.approx(+0.5)     # k_alt_p
    assert init[N + 1] == pytest.approx(-0.5)     # k_alt_d
    assert init[N + 2] == pytest.approx(+0.3)     # k_tilt
    assert init[N + 3] == pytest.approx(-0.15)    # k_rate
    assert init[N + 4] == pytest.approx(+0.2)     # k_yaw_rate


def test_default_init_params_scales_with_morph(hex_setup):
    """Supplying mass/inertia scales gains by the morph/ref ratio:
       k_alt_*    ∝ mass / REF_MASS
       k_tilt/_d  ∝ Iyy  / REF_IYY
       k_yaw_rate ∝ Izz  / REF_IZZ
    Verify by passing exactly the reference values (→ scaling = 1, so
    matches the unscaled warm-start) and then 2× values (→ 2× gains)."""
    _, _, prior = hex_setup
    N = prior.n_motors

    # Reference values → identity scaling → matches unscaled
    ref_inertia = np.diag([prior.REF_IYY, prior.REF_IYY, prior.REF_IZZ])
    init_ref = prior.default_init_params(mass=prior.REF_MASS, inertia=ref_inertia)
    init_plain = prior.default_init_params()
    np.testing.assert_allclose(init_ref, init_plain, rtol=1e-6)

    # 2× mass and inertia → 2× gains in every PD term
    init_2x = prior.default_init_params(
        mass=2 * prior.REF_MASS,
        inertia=np.diag([2 * prior.REF_IYY, 2 * prior.REF_IYY, 2 * prior.REF_IZZ]),
    )
    for i in range(5):
        assert init_2x[N + i] == pytest.approx(2.0 * init_plain[N + i], rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Shape handling — batched vs single-env
# ─────────────────────────────────────────────────────────────────────────────

def test_prior_effort_single_env_shape(hex_setup):
    _, _, prior = hex_setup
    state = torch.zeros(12 + prior.n_motors)
    params = torch.tensor(prior.default_init_params())
    out = prior.prior_effort(state, params)
    assert out.shape == (prior.n_motors,)


def test_prior_effort_batched_shape(hex_setup):
    _, _, prior = hex_setup
    B = 17
    state = torch.zeros(B, 12 + prior.n_motors)
    params = torch.tensor(prior.default_init_params()).expand(B, -1)
    out = prior.prior_effort(state, params)
    assert out.shape == (B, prior.n_motors)


def test_prior_action_in_action_range(hex_setup):
    _, _, prior = hex_setup
    state = torch.zeros(8, 12 + prior.n_motors)
    state[:, 2] = prior.target[2]                     # at hover altitude
    params = torch.tensor(prior.default_init_params()).expand(8, -1)
    a = prior.prior_action(state, params)
    assert a.shape == (8, prior.n_motors)
    assert (a >= -1.0).all() and (a <= 1.0).all()


def test_at_hover_target_effort_is_zero_except_yaw(hex_setup):
    """State exactly at hover with zero rates and zero attitude → all
    feedback terms are zero. Effort vector equals trim (=zeros by default).
    """
    _, _, prior = hex_setup
    state = torch.zeros(12 + prior.n_motors)
    state[2] = prior.target[2]            # at altitude
    params = torch.tensor(prior.default_init_params())
    eff = prior.prior_effort(state, params)
    np.testing.assert_allclose(eff.numpy(), np.zeros(prior.n_motors), atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# SIGN-CONVENTION GUARDS  (the load-bearing tests)
# These would have caught the original positive-feedback pitch bug. If
# they fail, do NOT weaken them — find the regression first.
# ─────────────────────────────────────────────────────────────────────────────

def _state_with_disturbance(prior: HoverPrior, **deltas) -> torch.Tensor:
    """Build a state at hover with named single-component deltas
    (keys: pitch, roll, yaw_rate, vz, z_err, pitch_rate, roll_rate)."""
    state = torch.zeros(12 + prior.n_motors)
    state[2] = prior.target[2]
    if "pitch" in deltas:      state[7]  = deltas["pitch"]
    if "roll" in deltas:       state[6]  = deltas["roll"]
    if "yaw_rate" in deltas:   state[11] = deltas["yaw_rate"]
    if "vz" in deltas:         state[5]  = deltas["vz"]
    if "z_err" in deltas:      state[2]  = prior.target[2] + deltas["z_err"]
    if "pitch_rate" in deltas: state[10] = deltas["pitch_rate"]
    if "roll_rate" in deltas:  state[9]  = deltas["roll_rate"]
    return state


def _step_open_loop(prior: HoverPrior, params, state, dyn, dt=0.01):
    """One Euler step using the prior's action and the analytical
    dynamics. Returns next state."""
    action = prior.prior_action(state, params)
    sd = dyn(state.unsqueeze(0).T, action.unsqueeze(0).T).T.squeeze(0)
    return state + dt * sd


def test_pitch_disturbance_commands_less_thrust_at_front(hex_setup):
    """Inject a positive pitch perturbation (nose up). Under the
    corrected mixer, the prior must command LESS thrust at motors with
    positive x (front of the body) and MORE at motors with negative x
    (rear). Tests the action vector directly to avoid the motor-lag /
    morph-asymmetry confounds of a one-step dynamics integration.

    This is the *exact* assertion that fails if the pitch mixer sign
    regresses to its original `+cos(phi)`."""
    morph, _, prior = hex_setup
    params = torch.tensor(prior.default_init_params())
    state = _state_with_disturbance(prior, pitch=+0.05)
    action = prior.prior_action(state, params)              # (N,)

    locs = np.array([p["loc"] for p in morph.propellers], dtype=np.float32)
    front_mask = locs[:, 0] > 0.01      # +x side
    rear_mask  = locs[:, 0] < -0.01     # -x side
    assert front_mask.any() and rear_mask.any(), (
        "morph has no clear front/rear motor split — pick a different fixture"
    )
    front_thrust = action[front_mask].mean().item()
    rear_thrust  = action[rear_mask].mean().item()
    assert front_thrust < rear_thrust, (
        f"nose-up disturbance should command less thrust at front "
        f"({front_thrust:+.4f}) than at rear ({rear_thrust:+.4f}). "
        f"Mixer sign bug suspected."
    )


def test_roll_disturbance_commands_more_thrust_on_low_side(hex_setup):
    """Positive roll (phi > 0) = right wing down in NED. Mixer should
    command MORE thrust on +y motors to lift the down side. Same logic
    as pitch test; this one defends the roll mixer column."""
    morph, _, prior = hex_setup
    params = torch.tensor(prior.default_init_params())
    state = _state_with_disturbance(prior, roll=+0.05)
    action = prior.prior_action(state, params)

    locs = np.array([p["loc"] for p in morph.propellers], dtype=np.float32)
    right_mask = locs[:, 1] > 0.01      # +y side
    left_mask  = locs[:, 1] < -0.01     # -y side
    assert right_mask.any() and left_mask.any()
    right_thrust = action[right_mask].mean().item()
    left_thrust  = action[left_mask].mean().item()
    assert right_thrust > left_thrust, (
        f"right-wing-down disturbance should command more thrust on the "
        f"right ({right_thrust:+.4f}) than the left ({left_thrust:+.4f})"
    )


def test_yaw_rate_disturbance_is_damped(hex_setup):
    """The new k_yaw_rate term must produce a torque that reduces |r|."""
    morph, params_dict, prior = hex_setup
    dyn = _build_torch_dynamics(
        params_dict, prior.n_motors, prior.gravity,
        prior.device, prior.dtype,
    )
    params = torch.tensor(prior.default_init_params())

    state0 = _state_with_disturbance(prior, yaw_rate=+0.3)   # 17°/s
    state1 = _step_open_loop(prior, params, state0, dyn)
    # r is state index 11
    abs_r_drops = abs(float(state1[11])) < abs(float(state0[11]))
    assert abs_r_drops, (
        f"Yaw rate should be damped; r went {state0[11].item():+.4f} "
        f"→ {state1[11].item():+.4f}"
    )


def test_below_target_altitude_pushes_up(hex_setup):
    """Drone too low (z_err < 0 in NED means above target since z is
    down-positive — careful with signs). 35c convention:
    z_err = state[2] - target[2], state[2] > target[2] means drone is
    LOWER (larger z = deeper), so positive z_err == too low.
    Under positive z_err, k_alt_p > 0, the alt_cmd > 0 → effort up →
    more thrust → drone rises (vz becomes more negative in NED).
    """
    morph, params_dict, prior = hex_setup
    dyn = _build_torch_dynamics(
        params_dict, prior.n_motors, prior.gravity,
        prior.device, prior.dtype,
    )
    params = torch.tensor(prior.default_init_params())

    state0 = _state_with_disturbance(prior, z_err=+0.3)   # 30 cm too low
    state1 = _step_open_loop(prior, params, state0, dyn)
    # vz should become more negative (drone climbing in NED)
    assert state1[5] < state0[5], (
        f"Drone below target should accelerate upward (vz↓). "
        f"vz went {state0[5].item():+.4f} → {state1[5].item():+.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Composability — the residual path (prior_effort + residual → action)
# ─────────────────────────────────────────────────────────────────────────────

def test_residual_path_equivalent_to_prior_action_when_residual_zero(hex_setup):
    """When the residual is zero, the composed action must equal the
    pure prior action. This is the contract Stage 2's residual env
    relies on — if it ever breaks, the residual training has implicit
    leakage."""
    _, _, prior = hex_setup
    state = torch.randn(4, 12 + prior.n_motors) * 0.1
    state[:, 2] = prior.target[2]
    params = torch.tensor(prior.default_init_params()).expand(4, -1)

    a_prior = prior.prior_action(state, params)
    a_composed = prior.effort_to_action(
        prior.prior_effort(state, params) + 0.0,
    )
    torch.testing.assert_close(a_prior, a_composed)


def test_residual_path_action_stays_in_range(hex_setup):
    """Even with a large residual, action remains in [-1, 1]."""
    _, _, prior = hex_setup
    state = torch.randn(4, 12 + prior.n_motors) * 0.2
    state[:, 2] = prior.target[2]
    params = torch.tensor(prior.default_init_params()).expand(4, -1)
    residual = torch.full((4, prior.n_motors), 5.0)   # huge

    a = prior.effort_to_action(prior.prior_effort(state, params) + 0.4 * residual)
    assert (a >= -1.0).all() and (a <= 1.0).all()
