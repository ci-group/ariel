"""Canonical analytical hover prior — single source of truth.

Imports
-------
This module owns the closed-form hover controller that we developed in
``35c_hover_cmaes_minimal.py`` (and verified end-to-end during the
sign-convention audit, see `IMPLEMENTATION_PLAN.md`). It is the
canonical implementation; **all consumers should import from here**:

  * `35c_hover_cmaes_minimal.py`  — training rollout
  * `35d_replay_cmaes_minimal.py` — replay
  * Stage 2 residual env (`envs/residual_drone_env.py`) — wraps this
    plus a learned residual

Keeping this in one place means the mixer-sign tests in
``test_prior_controller.py`` defend every consumer at once. The bug we
hunted during the hover work (positive-feedback pitch mixer) cannot
silently reappear in any one consumer.

Layout of the parameter vector
------------------------------
``cmaes_params`` is the 11d (N+5 for hex) vector that CMA-ES optimises:

  index range          name              meaning
  ─────────────────────────────────────────────────────────────────────
  [0:N]                trim              per-motor static thrust bias
  [N+0]                k_alt_p           P gain on altitude error (NED z)
  [N+1]                k_alt_d           D gain on vertical velocity
  [N+2]                k_tilt            P gain on roll/pitch angle
  [N+3]                k_rate            D gain on roll/pitch body rates
  [N+4]                k_yaw_rate        D gain on body yaw rate

State layout (matches ``torch_drone_gate_env._dynamics_body``)
--------------------------------------------------------------
  state[0:3]   = pos_ned   (x, y, z; z↓ positive)
  state[3:6]   = vel_world (vx, vy, vz)
  state[6:9]   = euler     (phi=roll, theta=pitch, psi=yaw)  ZYX intrinsic
  state[9:12]  = body_rate (p, q, r)
  state[12:]   = motor_w   (normalised motor speeds)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

GRAVITY_DEFAULT = 9.81
N_GAINS = 5  # k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate


# ─────────────────────────────────────────────────────────────────────────────
# Mixers — sign convention pinned (see docstrings)
# ─────────────────────────────────────────────────────────────────────────────

def tilt_mixer(propellers: Sequence[dict]) -> np.ndarray:
    """Per-motor mixer columns for [pitch_err, roll_err] feedback.

    Returns ``(N, 2)`` with column 0 = pitch contribution, column 1 = roll.

    Sign convention (verified against
    ``src/ariel/simulation/drone/dynamics_params.py::derive_reference_params``
    and the body dynamics in
    ``src/ariel/simulation/tasks/torch_drone_gate_env.py::_dynamics_body``):

      * **Roll**:  ``k_p_signed[i] = -y_i · k_f / Ixx``. For a right-side
        motor (+y), positive `phi` (right wing down in NED) needs MORE
        thrust to lift the down side. Mixer contribution is ``+sin(phi)``.
        ✓ matches the raw kinematic azimuth.
      * **Pitch**: ``k_q_signed[i] = +x_i · k_f / Iyy``. For a front
        motor (+x), positive `theta` (nose up in NED) needs LESS thrust
        to push the nose back down. Mixer contribution must be
        ``-cos(phi)``. The asymmetric sign on `y` vs `x` in
        ``dynamics_params`` is exactly why pitch and roll need opposite
        mixer signs despite both coming from the same kinematic azimuth.

    The original hand-derived mixer in `35c` had the pitch sign wrong,
    which silently caused CMA to drive ``k_tilt → 0`` (positive
    feedback on pitch error). Fix verified end-to-end:
    pre-fix `k_tilt ≈ 0`, post-fix `k_tilt ≈ +1.1`, hover time tripled.
    Sign tests in ``test_prior_controller.py`` guard against regression.
    """
    mix = np.zeros((len(propellers), 2), dtype=np.float32)
    for i, p in enumerate(propellers):
        loc = np.asarray(p["loc"], dtype=np.float32)
        phi = math.atan2(float(loc[1]), float(loc[0]))
        mix[i, 0] = -math.cos(phi)   # pitch  (note negative)
        mix[i, 1] =  math.sin(phi)   # roll
    return mix


def yaw_mixer(propellers: Sequence[dict]) -> np.ndarray:
    """Per-motor mixer for body-yaw-rate feedback.

    Returns ``(N,)``.

    Yaw torque in the dynamics is ``Mz = Σ k_r_signed[i] · W_i``, with
    ``k_r_signed[i] = spin_i · 2·k_m·W_hover / Izz`` (``spin_i = +1``
    for CCW, ``-1`` for CW). To damp positive body yaw rate ``r`` we
    need ``Mz < 0``, which means:
      * decrease ``W`` on CCW motors (spin = +1)
      * increase ``W`` on CW  motors (spin = -1)
    i.e. ``Δaction_i = -spin_i · k_yaw_rate · r``.

    We bake the minus sign into the mixer so the learned
    ``k_yaw_rate`` stays positive (matches the convention of
    ``tilt_mixer``).

    Drones with imbalanced spin (more CCW than CW or vice versa) build
    a static yaw torque at hover that ``trim`` cannot cancel on its
    own — trim biases thrust uniformly across spin sign whereas yaw
    needs a *differential*. The sampler in ``hex_sampler.py`` enforces
    balanced 3+3 spin, but this controller still works in the unbalanced
    case (it just compensates dynamically rather than statically).
    """
    mix = np.zeros(len(propellers), dtype=np.float32)
    for i, p in enumerate(propellers):
        spin = +1.0 if p["dir"][3] == "ccw" else -1.0
        mix[i] = -spin
    return mix


# ─────────────────────────────────────────────────────────────────────────────
# u_hover scalar (the analytical hover throttle)
# ─────────────────────────────────────────────────────────────────────────────

def compute_u_hover(params: dict, n_motors: int, gravity: float = GRAVITY_DEFAULT) -> float:
    """Closed-form hover-throttle scalar matching 35c.

    Given the dynamics-parameter dict from
    ``derive_reference_params``, returns the motor-command value in
    ``[-1, 1]`` that, applied uniformly to all motors, exactly cancels
    gravity at the linearisation point.
    """
    k_w, k_sq = float(params["k_w"]), float(params["k"])
    w_min, w_max = float(params["w_min"]), float(params["w_max"])
    w_hover = math.sqrt(gravity / (k_w * n_motors))
    z = float(np.clip((w_hover - w_min) / (w_max - w_min), 0.0, 1.0))
    disc = (1.0 - k_sq) ** 2 + 4.0 * k_sq * z * z
    u_hover_raw = (-(1.0 - k_sq) + math.sqrt(max(disc, 0.0))) / (2.0 * k_sq)
    return float(np.clip(2.0 * u_hover_raw - 1.0, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# HoverPrior — main class
# ─────────────────────────────────────────────────────────────────────────────

class HoverPrior:
    """Analytical morphology-conditional hover prior.

    Construct once per morphology (precomputes mixers, ``u_hover``,
    moves tensors to device). Apply many times with different state /
    cmaes_params batches.

    The class exposes the controller at two granularities:

      * `prior_effort(state, params)` — per-motor raw effort, the same
        quantity 35c sums up before its clamp/scale stage. **This is the
        right thing to compose with a residual policy**:

              effort = prior.prior_effort(state, params) + α · residual_action
              action = prior.effort_to_action(effort)

      * `prior_action(state, params)` — fully clamped/scaled motor
        command in [-1, 1], ready to feed to dynamics. Used by 35c
        (training rollouts) and 35d (replay) where there is no
        residual.

    All methods accept either a single sample ``(state_dim,) /
    (param_dim,)`` or a batch ``(B, state_dim) / (B, param_dim)`` and
    return matching shape.
    """

    def __init__(
        self,
        propellers: Sequence[dict],
        params: dict,
        target_ned: Sequence[float],
        *,
        gravity: float = GRAVITY_DEFAULT,
        action_scale: float = 0.4,
        twr: float | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        propellers
            List of propeller dicts (loc, dir, propsize). Determines the
            mixers and the motor count.
        params
            Output of
            ``ariel.simulation.drone.dynamics_params.derive_reference_params``
            for this morphology. Provides ``k_w``, ``w_min``, ``w_max``,
            ``k`` used by `compute_u_hover`.
        target_ned
            3-vector hover target in NED coordinates (z negative for
            altitude above ground).
        gravity
            m/s² (default 9.81).
        action_scale
            Multiplier applied to the inner-clamped effort before adding
            ``u_hover``. 35c uses 0.4; the residual env uses the same so
            that prior+residual share the same control-authority budget.
        twr
            If provided, scales `action_scale` by
            ``(TWR_REF / twr)**2`` (clamped ≤ 1). High-TWR morphs
            (prop-6/7 hexes) over-correct with the reference 0.4 and
            crash; this auto-reduces authority where needed.
        """
        self.n_motors = len(propellers)
        self.gravity = float(gravity)
        TWR_REF = 32.0
        if twr is not None and twr > TWR_REF:
            action_scale = float(action_scale) * (TWR_REF / float(twr)) ** 2
        self.action_scale = float(action_scale)
        self.device = torch.device(device)
        self.dtype = dtype

        # Mixers — cached tensors on the right device
        self.mix = torch.tensor(tilt_mixer(propellers), device=self.device, dtype=self.dtype)
        self.yaw_mix = torch.tensor(yaw_mixer(propellers), device=self.device, dtype=self.dtype)

        # Target and u_hover
        self.target = torch.tensor(list(target_ned), device=self.device, dtype=self.dtype)
        self.u_hover = compute_u_hover(params, self.n_motors, self.gravity)

        # Stash dynamics constants + motor positions for analytical trim.
        self._k_w = float(params["k_w"])
        self._k_sq = float(params["k"])
        self._w_min = float(params["w_min"])
        self._w_max = float(params["w_max"])
        self._motor_xy = np.asarray(
            [[float(p["loc"][0]), float(p["loc"][1])] for p in propellers],
            dtype=np.float32,
        )

    # ── shape constants ────────────────────────────────────────────────

    @property
    def param_dim(self) -> int:
        """N (trim) + 5 (gains) = N + 5."""
        return self.n_motors + N_GAINS

    @property
    def state_dim_min(self) -> int:
        """Minimum state width needed (the 12 rigid-body dims; motor-w
        dims beyond 12 are not used by the prior)."""
        return 12

    # ── parameter helpers ──────────────────────────────────────────────

    # Reference morph used to calibrate the hand-tuned gains in 35c.
    # Values approximate the canonical test hex (mass ≈ 0.20 kg,
    # Iyy ≈ 1.5e-3 kg·m², Izz ≈ 3.0e-3 kg·m²). Used for the inertia-
    # scaled warm-start in `default_init_params(mass=..., inertia=...)`.
    REF_MASS = 0.20            # kg
    REF_IYY = 1.5e-3           # kg·m²
    REF_IZZ = 3.0e-3           # kg·m²

    def default_init_params(
        self,
        *,
        mass: float | None = None,
        inertia: np.ndarray | None = None,
    ) -> np.ndarray:
        """Warm-started parameter vector matching 35c.

        Trims zeroed; gains hand-picked with the correct *signs* so CMA
        starts on the right side of zero.

        If `mass` and `inertia` are supplied, the gains are **scaled by
        the morph's mass / inertia** to keep the closed-loop bandwidth
        roughly constant across morphs (PD theory: for fixed natural
        frequency ω_n, ``k_p ∝ I`` and similar for translational
        ``k ∝ m``). Without this, library builds across diverse hex
        sizes get stuck — small-inertia morphs over-correct and crash
        instantly with the reference-sized gains, large-inertia morphs
        under-correct.

        Returns numpy so it can be fed directly into ``ng.p.Array(init=...)``.
        """
        init = np.zeros(self.param_dim, dtype=np.float32)
        N = self.n_motors

        # Scaling factors. Default to 1.0 if the caller didn't supply
        # morph properties (back-compat with the original 35c warm-start).
        s_mass = 1.0
        s_iyy = 1.0
        s_izz = 1.0
        if mass is not None:
            s_mass = float(mass) / self.REF_MASS
        if inertia is not None:
            inertia_arr = np.asarray(inertia)
            s_iyy = float(inertia_arr[1, 1]) / self.REF_IYY
            s_izz = float(inertia_arr[2, 2]) / self.REF_IZZ

        init[N + 0] =  0.5  * s_mass    # k_alt_p     (translational, ∝ mass)
        init[N + 1] = -0.5  * s_mass    # k_alt_d     (translational, ∝ mass)
        init[N + 2] =  0.3  * s_iyy     # k_tilt      (pitch/roll P, ∝ Iyy)
        init[N + 3] = -0.15 * s_iyy     # k_rate      (pitch/roll D, ∝ Iyy)
        init[N + 4] =  0.2  * s_izz     # k_yaw_rate  (yaw D, ∝ Izz)

        # Analytical trim: cancel static roll/pitch moment caused by
        # asymmetric arm magnitudes (the sampler's ±30% per-motor jitter).
        # Without this, low-inertia high-thrust morphs (prop=6/7 hexes)
        # flip in <0.1s before CMA can refine the gains.
        init[0:N] = self._analytical_trim().astype(np.float32)
        return init

    def _analytical_trim(self) -> np.ndarray:
        """Per-motor trim that nulls static roll & pitch moment at hover.

        Min-norm solution to
            sum(delta_r_i) = 0          (preserves Fz)
            sum(y_i delta_r_i) = -Σy_i  (zero roll moment)
            sum(x_i delta_r_i) = -Σx_i  (zero pitch moment)
        where ``r_i = F_i / F_hover``. Maps r_i → target W_i via
        ``F = k_w W²`` and inverts the throttle curve to get u_i, then
        ``trim_i = (u_i - u_hover) / action_scale``.
        """
        N = self.n_motors
        xy = self._motor_xy
        x = xy[:, 0].astype(np.float64)
        y = xy[:, 1].astype(np.float64)
        ones = np.ones(N, dtype=np.float64)
        A = np.stack([ones, y, x], axis=0)            # (3, N)
        b = np.array([0.0, -float(y.sum()), -float(x.sum())], dtype=np.float64)
        # Min-norm delta_r = A.T (A A.T)^-1 b
        try:
            G = A @ A.T
            delta_r = A.T @ np.linalg.solve(G, b)
        except np.linalg.LinAlgError:
            return np.zeros(N, dtype=np.float32)
        r = 1.0 + delta_r                              # target thrust ratios
        r = np.clip(r, 0.05, 5.0)                      # numerical safety

        # Map r_i → u_i. F_hover = m g / N is implicit; we work in W.
        w_hover = math.sqrt(self.gravity / (self._k_w * N))
        W = w_hover * np.sqrt(r)
        z = np.clip((W - self._w_min) / (self._w_max - self._w_min), 1e-6, 1.0)
        k = self._k_sq
        # u_raw solves k*u_raw^2 + (1-k)*u_raw - z^2 = 0
        disc = (1.0 - k) ** 2 + 4.0 * k * z * z
        u_raw = (-(1.0 - k) + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * k)
        u = np.clip(2.0 * u_raw - 1.0, -1.0, 1.0)
        # trim_i is in effort space (action_scale-relative)
        scale = max(self.action_scale, 1e-3)
        trim = (u - self.u_hover) / scale
        return np.clip(trim, -1.0, 1.0)

    # ── effort / action ────────────────────────────────────────────────

    def _split_params(
        self, params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice ``params`` into ``(trim, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate)``.
        Each gain is unsqueezed on the last axis so it broadcasts against
        ``(B, N)`` per-motor quantities."""
        N = self.n_motors
        trim       = params[..., 0:N]
        k_alt_p    = params[..., N + 0:N + 1]
        k_alt_d    = params[..., N + 1:N + 2]
        k_tilt     = params[..., N + 2:N + 3]
        k_rate     = params[..., N + 3:N + 4]
        k_yaw_rate = params[..., N + 4:N + 5]
        return trim, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate

    @torch.no_grad()
    def prior_effort(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Per-motor raw effort, before clamp/scale.

        Compose with a residual via::

            effort = prior.prior_effort(state, params) + α · residual_action
            action = prior.effort_to_action(effort)

        Parameters
        ----------
        state  : (B, 12+N) or (12+N,)
        params : (B, N+5) or (N+5,) — must match leading batch dim of state
                 (or be unsqueezed by 1 in the unbatched case).

        Returns
        -------
        Tensor of shape (B, N) or (N,) matching the leading dim of state.
        """
        batched = state.dim() == 2
        if not batched:
            state = state.unsqueeze(0)
            params = params.unsqueeze(0) if params.dim() == 1 else params

        # Slice state
        z_err      = state[:, 2:3] - self.target[2]
        vz         = state[:, 5:6]
        roll       = state[:, 6:7]
        pitch      = state[:, 7:8]
        roll_rate  = state[:, 9:10]
        pitch_rate = state[:, 10:11]
        yaw_rate   = state[:, 11:12]

        trim, k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate = self._split_params(params)

        alt_cmd  = k_alt_p * z_err - k_alt_d * vz                                       # (B, 1)
        att_cmd  = k_tilt * (self.mix[:, 0].unsqueeze(0) * pitch
                             + self.mix[:, 1].unsqueeze(0) * roll)                      # (B, N)
        rate_cmd = k_rate * (self.mix[:, 0].unsqueeze(0) * pitch_rate
                             + self.mix[:, 1].unsqueeze(0) * roll_rate)                 # (B, N)
        yaw_cmd  = k_yaw_rate * self.yaw_mix.unsqueeze(0) * yaw_rate                    # (B, N)

        effort = trim + alt_cmd + att_cmd + rate_cmd + yaw_cmd                          # (B, N)
        return effort if batched else effort.squeeze(0)

    def effort_to_action(self, effort: torch.Tensor) -> torch.Tensor:
        """35c's clamp+scale+u_hover transform.

        ``action = (u_hover + clamp(effort, ±1) · action_scale).clamp(±1)``
        """
        return (self.u_hover + effort.clamp(-1.0, 1.0) * self.action_scale).clamp(-1.0, 1.0)

    @torch.no_grad()
    def prior_action(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Convenience: fully composed prior action ready for dynamics.

        Equivalent to ``effort_to_action(prior_effort(state, params))``.
        """
        return self.effort_to_action(self.prior_effort(state, params))


__all__ = [
    "HoverPrior",
    "N_GAINS",
    "compute_u_hover",
    "tilt_mixer",
    "yaw_mixer",
]
