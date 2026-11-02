# Drone dynamics model

Short reference for the reference-form dynamics shared between
`DroneSimulator._setup_dynamics` (symbolic) and
`torch_drone_gate_env._dynamics_body` (torch). Per-step parity between the
two is tested at machine precision in
`tests/unit/test_simulation/test_tasks/test_torch_numpy_dynamics_equivalence.py`.

## State and action

State vector of length `12 + N` where `N` is the motor count:

| idx     | meaning                              | frame      |
|---------|--------------------------------------|------------|
| 0–2     | x, y, z position                     | world (NED)|
| 3–5     | vx, vy, vz velocity                  | world (NED)|
| 6–8     | roll (phi), pitch (theta), yaw (psi) | world ZYX  |
| 9–11    | p, q, r body rates                   | body       |
| 12..    | w_i motor state ∈ [-1, 1]            | normalised |

Normalisation: `W_i = (w_i + 1) · 0.5 · (W_MAX_N − W_MIN_N) + W_MIN_N`
with `W_MIN_N = 0`, `W_MAX_N = 3000` rad/s. This is intentionally
independent of the physical motor idle floor `w_min` (which only enters
the Wc sqrt-poly mapping below).

Action vector of length `N`: `u_i ∈ [-1, 1]`. Maps to throttle
`U_i = (u_i + 1) · 0.5 ∈ [0, 1]`.

## Motor model (first-order lag + sqrt-polynomial)

```
Wc_i = (w_max − w_min) · sqrt(k · U_i² + (1 − k) · U_i) + w_min
dW_i = (Wc_i − W_i) / tau
```

`Wc` is the steady-state motor speed the controller commands; `tau` is
the motor time constant. `k`, `w_min`, `w_max`, `tau` come from the
sysid'd prop-size data in `propeller_data.py`.

## Force and moment per motor

Each motor has body-frame position `r_i = (x_i, y_i, z_i)` and unit
thrust normal `n_i = (nx_i, ny_i, nz_i)` from
`blueprint_to_propellers(bp, convention="ned")`. For an upright motor,
`n_i = (0, 0, -1)` (thrust points along world −z when the drone is level).

Thrust force (body frame):

```
F_i = k_f · W_i² · n_i          (3-vector)
```

Moments on the body:

1. **Force × moment arm**: `r_i × F_i`.
2. **Spin reaction** (Newton 3rd law from spinning rotor), linearised at
   hover throttle so the yaw-torque coefficient becomes
   `-spin_i · 2 · k_m · W_hover` instead of the nonlinear `k_m · W_i²`.
   Linearisation introduced in `dynamics_params.py` to match the
   `optimal_quad_control_RL` reference sysid.
3. **Motor-speed-derivative reaction** from `dW_i` (small; kept for
   parity with reference).

All three sources are projected onto body (x, y, z) axes and divided by
the respective body inertia components. `k_r_react` from
`propeller_data.py` is already a sysid'd Izz-normalised constant; we
treat it as a direct angular-accel coefficient so the upright case
reproduces the previous scalar formula exactly.

## Per-motor coefficients returned by `derive_reference_params`

12 arrays of length N, consumed by both dynamics bodies:

```
# Translational accel (body frame)
k_fx[i] = k_f · nx_i / m
k_fy[i] = k_f · ny_i / m
k_fz[i] = k_f · nz_i / m

# Angular accel from thrust force × moment arm
k_Mx_force[i] = k_f · (y_i·nz_i − z_i·ny_i) / Ixx
k_My_force[i] = k_f · (z_i·nx_i − x_i·nz_i) / Iyy
k_Mz_force[i] = k_f · (x_i·ny_i − y_i·nx_i) / Izz

# Angular accel from spin reaction (linearised at hover)
k_Mx_react[i] = -spin_i · 2·k_m·W_hover · nx_i / Ixx
k_My_react[i] = -spin_i · 2·k_m·W_hover · ny_i / Iyy
k_Mz_react[i] = -spin_i · 2·k_m·W_hover · nz_i / Izz

# Angular accel from motor-speed-derivative reaction
k_Mx_dot[i] = -spin_i · k_r_react · nx_i
k_My_dot[i] = -spin_i · k_r_react · ny_i
k_Mz_dot[i] = -spin_i · k_r_react · nz_i
```

Dynamics body computes:

```
Fx_body = Σ k_fx[i] · W_i²        (same for Fy, Fz)
Dx_body = -k_x · v_bx · ΣW         (same for Dy; drag; v_b is body velocity)
Mx_body = Σ k_Mx_force[i] · W_i²  +  Σ k_Mx_react[i] · W_i  +  Σ k_Mx_dot[i] · dW_i
                                              (same shape for My, Mz)
accel_world = [0, 0, g] + R · [Fx + Dx, Fy + Dy, Fz]
rates̈_body  = (Mx, My, Mz)
```

For upright-motor configs (nx = ny = 0, nz = -1, z_i = 0) the coefficients
collapse exactly to the previous `(k_p, k_q, k_r, k_r_react)` quadruple;
parity is guarded by `test_dynamics_func_matches_torch_build`. Tilted
motors (non-trivial `n_i` or `z_i ≠ 0`) are additionally covered by
`test_dynamics_tilted_motor_matches`.

## Reference

The 22-parameter reference form comes from
`optimal_quad_control_RL/randomization.py:5-10` (params_5inch, sysid'd
from flight data). The pre-generalisation airevolve runtime assumed all
motors were upright and only used `(x_i, y_i)` from the morphology; the
`arm_elevation`, `motor_disc_azimuth`, and `motor_disc_pitch` genome
columns were effectively dead weight. The generalisation above makes
every column meaningful without losing per-step parity against the
reference for upright quads.
