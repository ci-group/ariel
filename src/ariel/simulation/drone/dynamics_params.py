"""Reference-form dynamics parameter derivation.

Maps an airevolve propeller configuration into the 22-parameter reference
dynamics form (matching optimal_quad_control_RL/randomization.py:5-10's
`params_5inch` schema, but with per-motor coefficients computed from the
actual airevolve morphology rather than sysid'd from flight data).

This is the airevolve runtime equivalent of
`experimentation/reference_drone_sim.py:derive_params_2inch_from_airevolve`.
The runtime version generalizes to any prop size via
`get_extended_prop_params` (Phase 1) and to any 4-motor symmetric or
asymmetric quad. Generalization to N motors arrives in Phase 4.

Sign convention: we bake position and spin signs into the per-motor
coefficients themselves, so the downstream symbolic build is
morphology-agnostic. The reference's `_build_dynamics_func` hardcodes signs
for a specific motor layout (see reference_drone_sim.py:280-291); our
runtime pulls those signs out of the formula and into the parameter dict.

* `k_p_signed[i] = -y_i · k_f / Ixx` so that `Mx = sum_i k_p_signed[i] · W_i²`.
* `k_q_signed[i] = +x_i · k_f / Iyy` so that `My = sum_i k_q_signed[i] · W_i²`.
* `k_r_signed[i] = spin_i · 2 · k_m · W_hover / Izz` for the linear-W yaw term.
* `k_r_react_signed[i] = spin_i · k_r_react_borrow` for the dW yaw term.

Where spin_i = +1 for "ccw", -1 for "cw" (the convention is verified via
the per-step parity test in unit_tests/test_dynamics_parity.py).

See `experimentation/RUNTIME_DYNAMICS_MIGRATION.md` Phase 2.1.
"""
from __future__ import annotations

import numpy as np

from .propeller_data import get_extended_prop_params


def _spin_sign(rotation: str) -> float:
    """+1 for ccw, -1 for cw. Consistent with the reference's convention
    (verified against optimal_quad_control_RL via the V0 parity test)."""
    if rotation == "ccw":
        return 1.0
    if rotation == "cw":
        return -1.0
    raise ValueError(f"unknown rotation direction: {rotation!r}")


def derive_reference_params(
    propellers: list,
    mass: float,
    inertia: np.ndarray,
    prop_size,
    gravity: float = 9.81,
) -> dict:
    """Derive a reference-form parameter dict for an airevolve drone config.

    Args:
        propellers: list of propeller dicts (`loc`, `dir`, `propsize`).
        mass: total drone mass in kg (from DroneConfiguration.mass).
        inertia: 3x3 inertia matrix in body frame.
        prop_size: prop size for fetching aerodynamic constants
            (k_drag, k_r_react, k, w_min). All propellers assumed same size.
        gravity: m/s².

    Returns:
        dict with keys:
            n_motors (int): number of motors
            k_w (float): thrust acceleration coefficient (a_z = -k_w · sum(W²))
            k_x, k_y (float): body-frame drag accel coefficients (single values, all motors)
            k_p_signed (list[float]): per-motor roll-moment coefficient (signed)
            k_q_signed (list[float]): per-motor pitch-moment coefficient (signed)
            k_r_signed (list[float]): per-motor yaw-moment coefficient (linear W, signed)
            k_r_react_signed (list[float]): per-motor yaw-moment from dW (signed)
            tau, k, w_min, w_max (float): motor model parameters

    Notes:
        * Asymmetric morphologies are supported via per-motor `k_p_i, k_q_i`
          computed from the actual `loc`. Spin asymmetry is supported via
          per-motor `k_r_signed`.
        * `k_x, k_y, k, w_min, k_r_react, tau` are single per-prop-size values
          (borrowed from the closest sysid set in propeller_data.py). Per-motor
          variation in these is a Phase 6 polish item.
        * `k_r_signed` is a *linearization* at hover throttle. For hover,
          `dMz/dW ≈ 2·k_m·W_hover`. Away from hover this loses fidelity; the
          reference accepts this as a sysid-level approximation.
    """
    n = len(propellers)
    if n == 0:
        raise ValueError("derive_reference_params: no propellers in config")

    extended = get_extended_prop_params(prop_size)
    k_f, k_m = extended["constants"]
    w_max = float(extended["wmax"])

    Ixx = float(inertia[0, 0])
    Iyy = float(inertia[1, 1])
    Izz = float(inertia[2, 2])
    m = float(mass)

    # Hover motor speed (per motor) for linearizing the yaw torque term.
    F_hover_per_motor = m * gravity / n
    if F_hover_per_motor <= 0 or k_f <= 0:
        raise ValueError(
            f"derive_reference_params: invalid hover-thrust calc "
            f"(F_hover={F_hover_per_motor}, k_f={k_f})"
        )
    W_hover = float(np.sqrt(F_hover_per_motor / k_f))

    k_p_signed = []
    k_q_signed = []
    k_r_signed = []
    k_r_react_signed = []
    for prop in propellers:
        x_i = float(prop["loc"][0])
        y_i = float(prop["loc"][1])
        spin = _spin_sign(prop["dir"][3])

        # M_x = sum_i (-y_i · F_z_i) / Ixx = sum_i (-y_i · k_f · W_i²) / Ixx
        k_p_signed.append(-y_i * k_f / Ixx)
        # M_y = sum_i (+x_i · F_z_i) / Iyy = sum_i (+x_i · k_f · W_i²) / Iyy
        k_q_signed.append(+x_i * k_f / Iyy)
        # M_z (steady, linearized at hover): spin_i · 2 · k_m · W_hover / Izz
        k_r_signed.append(spin * 2.0 * k_m * W_hover / Izz)
        # M_z (motor-acceleration reaction): spin_i · k_r_react
        k_r_react_signed.append(spin * float(extended["k_r_react"]))

    return {
        "n_motors": n,
        "k_w": k_f / m,
        "k_x": float(extended["k_x_drag"]),
        "k_y": float(extended["k_y_drag"]),
        "k_p_signed": k_p_signed,
        "k_q_signed": k_q_signed,
        "k_r_signed": k_r_signed,
        "k_r_react_signed": k_r_react_signed,
        "tau": float(extended["tau"]),
        "k": float(extended["k"]),
        "w_min": float(extended["w_min"]),
        "w_max": w_max,
    }


# Reference's normalization constants (verified against
# optimal_quad_control_RL/quad_race_env.py:41-42 via the V0 parity test).
# These are intentionally independent of physical w_max; the motor state
# `w_i ∈ [-1, 1]` represents `W_i ∈ [W_MIN_N, W_MAX_N] = [0, 3000]` rad/s.
# At full throttle, Wc may exceed W_MAX_N; the state can go outside [-1, 1]
# during transients. This is intentional in the reference.
W_MIN_N = 0.0
W_MAX_N = 3000.0
