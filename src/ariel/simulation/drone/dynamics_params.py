"""Reference-form dynamics parameter derivation.

Maps an airevolve propeller configuration into a per-motor coefficient set
consumed by both the symbolic (``DroneSimulator._setup_dynamics``) and the
PyTorch (``torch_drone_gate_env._dynamics_body``) dynamics bodies.

The dynamics assume each motor i has position ``r_i = (x_i, y_i, z_i)`` and
unit thrust normal ``n_i = (nx_i, ny_i, nz_i)`` in the body frame. Thrust is
``F_i = k_f · W_i² · n_i`` (a 3-vector, not just along -z). This generalises
the previous "all motors upright" assumption so morphologies with tilted
propellers and non-flat arm layouts are actually simulated by the dynamics
rather than silently flattened.

Coefficients returned by :func:`derive_reference_params`
----------------------------------------------------------

Shared (one per morphology):
  * ``n_motors``: number of motors
  * ``k_w``: scalar k_f/m, kept for API compat (``T = -k_w · ΣW²`` in the
    legacy formulation; new code should prefer ``k_fx/fy/fz`` instead)
  * ``k_x, k_y``: body-frame drag coefficients
  * ``tau, k, w_min, w_max``: motor model parameters

Per-motor arrays (length n_motors) — force / translational accel:
  * ``k_fx[i] = k_f · nx_i / m``   contributes to body-frame a_x
  * ``k_fy[i] = k_f · ny_i / m``   contributes to body-frame a_y
  * ``k_fz[i] = k_f · nz_i / m``   contributes to body-frame a_z

Per-motor arrays — moment from thrust force (r × F):
  * ``k_Mx_force[i] = k_f · (y_i·nz_i − z_i·ny_i) / Ixx``
  * ``k_My_force[i] = k_f · (z_i·nx_i − x_i·nz_i) / Iyy``
  * ``k_Mz_force[i] = k_f · (x_i·ny_i − y_i·nx_i) / Izz``

Per-motor arrays — moment from spin reaction (linearised at hover W):
  * ``k_Mx_react[i] = -spin_i · 2·k_m·W_hover · nx_i / Ixx``
  * ``k_My_react[i] = -spin_i · 2·k_m·W_hover · ny_i / Iyy``
  * ``k_Mz_react[i] = -spin_i · 2·k_m·W_hover · nz_i / Izz``

Per-motor arrays — moment from motor acceleration (dW) reaction:
  * ``k_Mx_dot[i] = -spin_i · k_r_react · nx_i``
  * ``k_My_dot[i] = -spin_i · k_r_react · ny_i``
  * ``k_Mz_dot[i] = -spin_i · k_r_react · nz_i``

For the upright case (``n_i = (0, 0, -1)``, ``z_i = 0``) these reduce exactly
to the previous ``k_p_signed, k_q_signed, k_r_signed, k_r_react_signed``
quadruple — see the parity test in
``tests/unit/test_simulation/test_tasks/test_torch_numpy_dynamics_equivalence.py``.

Thrust normals come from ``blueprint_to_propellers(bp, convention="ned")``
so ``n_i`` is a unit vector in NED body frame (``nz_i ≈ -1`` for upright
motors; thrust acts along -z world when the drone is level).
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
    """Derive the per-motor coefficient dict for an airevolve drone config.

    Args:
        propellers: list of propeller dicts with ``loc = (x, y, z)``,
            ``dir = (nx, ny, nz, rotation)``, and ``propsize``. ``n`` is
            expected to be a unit vector (as emitted by
            :func:`ariel.body_phenotypes.drone.backends.blueprint_to_propellers`).
        mass: total drone mass in kg (from ``DroneConfiguration.mass``).
        inertia: 3x3 inertia matrix in body frame.
        prop_size: prop size for fetching aerodynamic constants
            (``k_drag``, ``k_r_react``, ``k``, ``w_min``). All propellers
            assumed to share the same size.
        gravity: m/s².

    Returns:
        dict with the coefficients documented at the top of this module.

    Notes:
        * Asymmetric morphologies and tilted motors are both supported.
        * ``k_x, k_y, k, w_min, k_r_react, tau`` are single per-prop-size
          values (borrowed from the closest sysid set in
          :mod:`propeller_data`). Per-motor variation is Phase 6 work.
        * The react term is a *linearisation* at hover throttle. For hover,
          ``dMz/dW ≈ 2·k_m·W_hover``. Away from hover this loses fidelity;
          the reference accepts this as a sysid-level approximation.
    """
    n = len(propellers)
    if n == 0:
        raise ValueError("derive_reference_params: no propellers in config")

    extended = get_extended_prop_params(prop_size)
    k_f, k_m = extended["constants"]
    w_max = float(extended["wmax"])
    k_r_react = float(extended["k_r_react"])

    Ixx = float(inertia[0, 0])
    Iyy = float(inertia[1, 1])
    Izz = float(inertia[2, 2])
    m = float(mass)

    F_hover_per_motor = m * gravity / n
    if F_hover_per_motor <= 0 or k_f <= 0:
        raise ValueError(
            f"derive_reference_params: invalid hover-thrust calc "
            f"(F_hover={F_hover_per_motor}, k_f={k_f})"
        )
    W_hover = float(np.sqrt(F_hover_per_motor / k_f))

    k_fx:        list[float] = []
    k_fy:        list[float] = []
    k_fz:        list[float] = []
    k_Mx_force:  list[float] = []
    k_My_force:  list[float] = []
    k_Mz_force:  list[float] = []
    k_Mx_react:  list[float] = []
    k_My_react:  list[float] = []
    k_Mz_react:  list[float] = []
    k_Mx_dot:    list[float] = []
    k_My_dot:    list[float] = []
    k_Mz_dot:    list[float] = []
    motor_positions: list[list[float]] = []
    thrust_normals:  list[list[float]] = []

    for prop in propellers:
        x_i, y_i, z_i = (float(v) for v in prop["loc"])
        nx_i, ny_i, nz_i, rotation = prop["dir"]
        nx_i = float(nx_i); ny_i = float(ny_i); nz_i = float(nz_i)
        spin = _spin_sign(rotation)

        # Translational accel from this motor's thrust (body frame).
        k_fx.append(k_f * nx_i / m)
        k_fy.append(k_f * ny_i / m)
        k_fz.append(k_f * nz_i / m)

        # Angular accel from thrust-force cross moment arm (r × F / I).
        k_Mx_force.append(k_f * (y_i * nz_i - z_i * ny_i) / Ixx)
        k_My_force.append(k_f * (z_i * nx_i - x_i * nz_i) / Iyy)
        k_Mz_force.append(k_f * (x_i * ny_i - y_i * nx_i) / Izz)

        # Angular accel from spin-reaction torque, linearised at hover W.
        # tau_react = -spin · 2·k_m·W_hover · W · n_i / I_axis
        k_Mx_react.append(-spin * 2.0 * k_m * W_hover * nx_i / Ixx)
        k_My_react.append(-spin * 2.0 * k_m * W_hover * ny_i / Iyy)
        k_Mz_react.append(-spin * 2.0 * k_m * W_hover * nz_i / Izz)

        # Angular accel from motor-speed-derivative reaction. k_r_react is a
        # sysid'd Izz-normalised constant from the reference; we treat it as
        # a per-axis angular-accel coefficient along n_i so the upright case
        # (nz=-1) reproduces the previous scalar formula exactly.
        k_Mx_dot.append(-spin * k_r_react * nx_i)
        k_My_dot.append(-spin * k_r_react * ny_i)
        k_Mz_dot.append(-spin * k_r_react * nz_i)

        motor_positions.append([x_i, y_i, z_i])
        thrust_normals.append([nx_i, ny_i, nz_i])

    return {
        "n_motors": n,
        "k_w": k_f / m,
        "k_x": float(extended["k_x_drag"]),
        "k_y": float(extended["k_y_drag"]),
        # Per-motor force / moment coefficients (12 arrays × n_motors).
        "k_fx": k_fx,
        "k_fy": k_fy,
        "k_fz": k_fz,
        "k_Mx_force": k_Mx_force,
        "k_My_force": k_My_force,
        "k_Mz_force": k_Mz_force,
        "k_Mx_react": k_Mx_react,
        "k_My_react": k_My_react,
        "k_Mz_react": k_Mz_react,
        "k_Mx_dot":   k_Mx_dot,
        "k_My_dot":   k_My_dot,
        "k_Mz_dot":   k_Mz_dot,
        # Motor model (shared).
        "tau": float(extended["tau"]),
        "k": float(extended["k"]),
        "w_min": float(extended["w_min"]),
        "w_max": w_max,
        # Debug / introspection — the raw motor placements and normals used.
        "motor_positions": motor_positions,
        "thrust_normals":  thrust_normals,
        # Underlying constants, kept for consumers that need to reconstruct.
        "k_f": float(k_f),
        "k_m": float(k_m),
        "m":   m,
        "W_hover": W_hover,
    }


# Reference's normalization constants (verified against
# optimal_quad_control_RL/quad_race_env.py:41-42 via the V0 parity test).
# These are intentionally independent of physical w_max; the motor state
# `w_i ∈ [-1, 1]` represents `W_i ∈ [W_MIN_N, W_MAX_N] = [0, 3000]` rad/s.
# At full throttle, Wc may exceed W_MAX_N; the state can go outside [-1, 1]
# during transients. This is intentional in the reference.
W_MIN_N = 0.0
W_MAX_N = 3000.0
