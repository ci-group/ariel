"""GPU-accelerated drone gate environment.

Replaces DroneGateEnv's SymPy-lambdified NumPy dynamics with a hand-written
PyTorch implementation so the full physics simulation runs on the target
device (CPU or CUDA).  All internal state is stored as torch.Tensor; only
the VecEnv boundary (observations, rewards, dones) converts back to NumPy
as required by Stable-Baselines3.

Drop-in replacement for DroneGateEnv — same constructor signature.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from ariel.simulation.drone.drone_simulator import DroneSimulator
from ariel.simulation.drone.dynamics_params import W_MAX_N, W_MIN_N

# ---------------------------------------------------------------------------
# Default gate track (matches DroneGateEnv)
# ---------------------------------------------------------------------------
_r = 1.5
_DEFAULT_GATE_POS = np.array([
    [ _r, -_r, -1.5], [0,    0, -1.5], [-_r,  _r, -1.5], [0,  2*_r, -1.5],
    [ _r,  _r, -1.5], [0,    0, -1.5], [-_r, -_r, -1.5], [0, -2*_r, -1.5],
], dtype=np.float32)
_DEFAULT_GATE_YAW  = np.array([1, 2, 1, 0, -1, -2, -1, 0], dtype=np.float32) * (math.pi / 2)
_DEFAULT_START_POS = (_DEFAULT_GATE_POS[0] + np.array([0, -1.0, 0])).astype(np.float32)

_W_RANGE = float(W_MAX_N - W_MIN_N)   # 3000.0


# ---------------------------------------------------------------------------
# PyTorch dynamics — module-level compiled function
# ---------------------------------------------------------------------------
# The dynamics body is a plain module-level function so torch.compile sees
# the same Python function object on every call, regardless of which
# morphology (individual) is being evaluated.  Per-morphology parameters
# (k_w, k_p, …) are passed as explicit tensor arguments rather than being
# captured in a per-individual closure; this means the compiled graph is
# traced once and reused across all individuals with the same motor count.

def _dynamics_body(
    state:  torch.Tensor,   # (12+N, E)
    action: torch.Tensor,   # (N, E)
    # ---- scalar params (0-d tensors, morphology-specific) ----
    k_w:  torch.Tensor,
    k_x:  torch.Tensor,
    k_y:  torch.Tensor,
    tau:  torch.Tensor,
    k_sq: torch.Tensor,
    w_lo: torch.Tensor,
    w_hi: torch.Tensor,
    g:    torch.Tensor,
    W_R:  torch.Tensor,
    # ---- per-motor params ((N,1) tensors) ----
    k_p:  torch.Tensor,
    k_q:  torch.Tensor,
    k_r:  torch.Tensor,
    k_rr: torch.Tensor,
) -> torch.Tensor:
    """Vectorised drone dynamics — state_dot = f(state, action, params).

    state  : (12+N, E)  — columns are parallel environments
    action : (N,    E)
    returns: (12+N, E)

    All parameters are plain tensors so this function can be compiled once
    and called with different per-morphology tensors without re-tracing.
    """
    # ---- unpack state ----------------------------------------------------
    vx, vy, vz      = state[3],  state[4],  state[5]
    phi, theta, psi = state[6],  state[7],  state[8]
    p,  q,  r       = state[9],  state[10], state[11]
    w               = state[12:]   # (N, E)

    # ---- trig ------------------------------------------------------------
    cphi, sphi = phi.cos(),   phi.sin()
    cth,  sth  = theta.cos(), theta.sin()
    cpsi, spsi = psi.cos(),   psi.sin()
    tanth       = sth / cth   # singular at ±π/2; OOB resets guard this

    # ---- rotation matrix R = Rz·Ry·Rx  (body → world) ------------------
    R00 = cpsi * cth
    R01 = cpsi * sth * sphi - spsi * cphi
    R02 = cpsi * sth * cphi + spsi * sphi
    R10 = spsi * cth
    R11 = spsi * sth * sphi + cpsi * cphi
    R12 = spsi * sth * cphi - cpsi * sphi
    R20 = -sth
    R21 = cth * sphi
    R22 = cth * cphi

    # ---- body-frame velocity for aerodynamic drag -----------------------
    vbx = R00 * vx + R10 * vy + R20 * vz
    vby = R01 * vx + R11 * vy + R21 * vz

    # ---- motor model ----------------------------------------------------
    W   = (w + 1.0) * (0.5 * W_R) + w_lo              # (N, E)  rad/s
    U   = ((action + 1.0) * 0.5).clamp(0.0, 1.0)      # (N, E)  ∈ [0,1]
    sq_arg = (k_sq * U * U + (1.0 - k_sq) * U).clamp(min=0.0)
    Wc  = (w_hi - w_lo) * sq_arg.sqrt() + w_lo        # (N, E)
    dW  = (Wc - W) / tau                               # (N, E)  rad/s²
    dw  = dW * (2.0 / W_R)                            # normalised (/s)

    # ---- aggregate motor quantities -------------------------------------
    W2     = W * W
    sum_W  = W.sum(dim=0)
    sum_W2 = W2.sum(dim=0)

    # ---- forces ---------------------------------------------------------
    T  = -k_w * sum_W2
    Dx = -k_x * vbx * sum_W
    Dy = -k_y * vby * sum_W

    # ---- moments --------------------------------------------------------
    Mx = (k_p  * W2).sum(dim=0)
    My = (k_q  * W2).sum(dim=0)
    Mz = (k_r  * W ).sum(dim=0) + (k_rr * dW).sum(dim=0)

    # ---- translational kinematics / dynamics ----------------------------
    d_x,  d_y,  d_z  = vx, vy, vz
    d_vx = R00 * Dx + R01 * Dy + R02 * T
    d_vy = R10 * Dx + R11 * Dy + R12 * T
    d_vz = g   + R20 * Dx + R21 * Dy + R22 * T

    # ---- Euler-angle kinematics (ZYX) -----------------------------------
    d_phi   = p + (q * sphi + r * cphi) * tanth
    d_theta = q * cphi - r * sphi
    d_psi   = (q * sphi + r * cphi) / cth

    # ---- rotational dynamics --------------------------------------------
    d_p, d_q, d_r_dot = Mx, My, Mz

    # ---- assemble -------------------------------------------------------
    base = torch.stack(
        [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r_dot],
        dim=0,
    )  # (12, E)
    return torch.cat([base, dw], dim=0)   # (12+N, E)


# Compiled once at import time for CUDA.  CPU path skips compilation because
# torch.compile on CPU offers no benefit for this workload.
_dynamics_body_compiled = torch.compile(_dynamics_body, mode="reduce-overhead")


def _build_torch_dynamics(
    params: dict[str, Any],
    num_motors: int,
    gravity: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Return a vectorised dynamics callable bound to this morphology's params.

    The returned function has the same signature as before::

        state_dot = dynamics(state, action)   # (12+N, E), (N, E) → (12+N, E)

    Internally it calls the module-level ``_dynamics_body`` (or its compiled
    variant on CUDA) with the per-morphology parameter tensors already bound,
    so torch.compile only traces once per motor-count regardless of how many
    individuals are evaluated.
    """
    def _t(v):
        return torch.tensor(v, device=device, dtype=dtype)

    # Build parameter tensors for this morphology.
    p_k_w  = _t(params["k_w"])
    p_k_x  = _t(params["k_x"])
    p_k_y  = _t(params["k_y"])
    p_tau  = _t(params["tau"])
    p_k_sq = _t(params["k"])
    p_w_lo = _t(params["w_min"])
    p_w_hi = _t(params["w_max"])
    p_g    = _t(gravity)
    p_W_R  = _t(_W_RANGE)
    p_k_p  = _t(params["k_p_signed"]).view(num_motors, 1)
    p_k_q  = _t(params["k_q_signed"]).view(num_motors, 1)
    p_k_r  = _t(params["k_r_signed"]).view(num_motors, 1)
    p_k_rr = _t(params["k_r_react_signed"]).view(num_motors, 1)

    # Choose compiled vs. plain depending on device.
    _fn = _dynamics_body_compiled if device.type == "cuda" else _dynamics_body

    def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return _fn(
            state, action,
            p_k_w, p_k_x, p_k_y, p_tau, p_k_sq, p_w_lo, p_w_hi, p_g, p_W_R,
            p_k_p, p_k_q, p_k_r, p_k_rr,
        )

    return dynamics


# ---------------------------------------------------------------------------
# TorchDroneGateEnv
# ---------------------------------------------------------------------------

class TorchDroneGateEnv(VecEnv):
    """DroneGateEnv with PyTorch dynamics for GPU-accelerated simulation.

    All env state lives on ``device`` as torch tensors.  NumPy conversion
    only happens at the SB3 VecEnv boundary (reset/step returns).

    Constructor signature is identical to DroneGateEnv so it is a drop-in
    replacement — just swap the class name.
    """

    metadata   = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(
        self,
        num_envs: int,
        propellers=None,
        individual=None,
        gates_pos=None,
        gate_yaw=None,
        start_pos=None,
        x_bounds=(-5, 5),
        y_bounds=(-5, 5),
        z_bounds=(-5, 5),
        gates_ahead: int = 2,
        motor_limit: float = 1.0,
        initialize_at_random_gates: bool = True,
        seed=None,
        render_mode=None,
        device="cpu",
        dt: float = 0.01,
        max_steps: int = 1200,
        action_filter_alpha: float = 1.0,
        gate_reward: float = 1.0,
        # Accepted for API compat with DroneGateEnv; not implemented here
        pause_if_collision: bool = False,
        num_state_history: int = 0,
        num_action_history: int = 0,
        history_step_size: int = 1,
    ) -> None:
        if num_state_history or num_action_history:
            raise NotImplementedError(
                "TorchDroneGateEnv does not support state/action history yet."
            )

        self.dev   = torch.device(device)
        self.dtype = torch.float32

        if render_mode is not None:
            self.render_mode = render_mode

        # Seeding
        self.seed_val = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Build simulator (params only — dynamics handled by _dynamics below)
        if propellers is not None:
            self.drone_sim = DroneSimulator(propellers=propellers, dt=dt)
        elif individual is not None:
            from ariel.simulation.tasks.drone_gate_env import DroneGateEnv as _DGE
            props, mounts = _DGE._convert_individual_to_propellers(None, individual)
            self.drone_sim = DroneSimulator(propellers=props, dt=dt)
        else:
            self.drone_sim = DroneSimulator.create_standard_drone("quad", dt=dt)

        self.num_motors = self.drone_sim.num_motors

        # Build the dynamics callable for this morphology's parameters.
        # The underlying _dynamics_body_compiled is a module-level compiled
        # function shared across all individuals, so torch.compile traces
        # only once per motor-count rather than once per individual.
        self._dynamics = _build_torch_dynamics(
            self.drone_sim.params, self.num_motors,
            self.drone_sim.g, self.dev, self.dtype,
        )

        # ---- gate track ------------------------------------------------
        _gpos = np.asarray(gates_pos  if gates_pos  is not None else _DEFAULT_GATE_POS, dtype=np.float32)
        _gyaw = np.asarray(gate_yaw   if gate_yaw   is not None else _DEFAULT_GATE_YAW, dtype=np.float32)
        _spos = np.asarray(start_pos  if start_pos  is not None else _DEFAULT_START_POS, dtype=np.float32)

        self.num_gates   = len(_gpos)
        self.gates_ahead = gates_ahead
        self.gate_size   = 1.5

        self.gate_pos_t  = torch.tensor(_gpos, device=self.dev, dtype=self.dtype)   # (G, 3)
        self.gate_yaw_t  = torch.tensor(_gyaw, device=self.dev, dtype=self.dtype)   # (G,)
        self.start_pos_t = torch.tensor(_spos, device=self.dev, dtype=self.dtype)   # (3,)

        # Precompute relative gate offsets for the observation vector
        gpr = np.zeros((self.num_gates, 3), dtype=np.float32)
        gyr = np.zeros(self.num_gates,       dtype=np.float32)
        for i in range(self.num_gates):
            gpr[i] = _gpos[i] - _gpos[i - 1]
            R2 = np.array([
                [ np.cos(_gyaw[i - 1]), np.sin(_gyaw[i - 1])],
                [-np.sin(_gyaw[i - 1]), np.cos(_gyaw[i - 1])],
            ])
            gpr[i, 0:2] = R2 @ gpr[i, 0:2]
            dy = _gyaw[i] - _gyaw[i - 1]
            gyr[i] = (dy + math.pi) % (2 * math.pi) - math.pi

        self.gate_pos_rel_t = torch.tensor(gpr, device=self.dev, dtype=self.dtype)   # (G, 3)
        self.gate_yaw_rel_t = torch.tensor(gyr, device=self.dev, dtype=self.dtype)   # (G,)

        # ---- env settings ----------------------------------------------
        self.initialize_at_random_gates = initialize_at_random_gates
        self.motor_limit        = motor_limit
        self.dt                 = float(dt)
        self.max_steps          = int(max_steps)
        self.action_filter_alpha = float(action_filter_alpha)
        self.gate_reward = float(gate_reward)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

        # ---- spaces ----------------------------------------------------
        n = self.num_motors
        u_lim = 2.0 * motor_limit - 1.0
        action_space      = spaces.Box(low=-1.0, high=u_lim, shape=(n,), dtype=np.float64)
        self.state_len    = 12 + n + 4 * gates_ahead
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_len,), dtype=np.float64
        )
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        # ---- GPU state tensors -----------------------------------------
        self.world_states    = torch.zeros((num_envs, 12 + n),         device=self.dev, dtype=self.dtype)
        self._obs_t          = torch.zeros((num_envs, self.state_len),  device=self.dev, dtype=self.dtype)
        self.target_gates    = torch.zeros(num_envs, device=self.dev, dtype=torch.long)
        self.step_counts     = torch.zeros(num_envs, device=self.dev, dtype=torch.long)
        self.num_gates_passed= torch.zeros(num_envs, device=self.dev, dtype=torch.long)
        self.actions_t       = torch.zeros((num_envs, n), device=self.dev, dtype=self.dtype)
        self.prev_actions_t  = torch.zeros((num_envs, n), device=self.dev, dtype=self.dtype)
        self.filtered_acts   = torch.zeros((num_envs, n), device=self.dev, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_obs(self) -> None:
        """Compute gate-relative observations into self._obs_t (all on device)."""
        tg      = self.target_gates % self.num_gates          # (E,)
        gpos    = self.gate_pos_t[tg]                         # (E, 3)
        gyaw    = self.gate_yaw_t[tg]                         # (E,)
        cg, sg  = gyaw.cos(), gyaw.sin()

        # Position relative to gate, rotated into gate frame
        dxy     = self.world_states[:, 0:2] - gpos[:, 0:2]   # (E, 2)
        obs     = self._obs_t
        obs[:, 0] =  cg * dxy[:, 0] + sg * dxy[:, 1]
        obs[:, 1] = -sg * dxy[:, 0] + cg * dxy[:, 1]
        obs[:, 2] = self.world_states[:, 2] - gpos[:, 2]

        # Velocity rotated into gate frame
        vel = self.world_states[:, 3:6]                       # (E, 3)
        obs[:, 3] =  cg * vel[:, 0] + sg * vel[:, 1]
        obs[:, 4] = -sg * vel[:, 0] + cg * vel[:, 1]
        obs[:, 5] = vel[:, 2]

        # Attitude (roll, pitch); yaw relative to gate
        obs[:, 6] = self.world_states[:, 6]
        obs[:, 7] = self.world_states[:, 7]
        yaw_rel   = self.world_states[:, 8] - gyaw
        obs[:, 8] = ((yaw_rel + math.pi) % (2 * math.pi)) - math.pi

        # Body rates and motor speeds
        obs[:, 9:12]              = self.world_states[:, 9:12]
        obs[:, 12:12+self.num_motors] = self.world_states[:, 12:12+self.num_motors]

        # Future gate offsets
        base = 12 + self.num_motors
        for i in range(self.gates_ahead):
            idx = (self.target_gates + i + 1) % self.num_gates
            obs[:, base+4*i : base+4*i+3] = self.gate_pos_rel_t[idx]
            obs[:, base+4*i+3]            = self.gate_yaw_rel_t[idx]

    def _reset_envs(self, mask: torch.Tensor) -> None:
        """Overwrite world state for envs where mask is True.

        Does NOT update _obs_t — caller must call _update_obs() afterwards.
        """
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return

        if self.initialize_at_random_gates:
            gi   = torch.randint(0, self.num_gates, (n_reset,), device=self.dev)
            self.target_gates[mask] = gi
            pg   = self.gate_pos_t[gi]                                # (R, 3)
            yg   = self.gate_yaw_t[gi]                                # (R,)
            pos  = pg - torch.stack([yg.cos(), yg.sin(),
                                     torch.zeros_like(yg)], dim=1)
            vel  = (torch.rand((n_reset, 3), device=self.dev, dtype=self.dtype) - 0.5)
            ang  = torch.zeros((n_reset, 3), device=self.dev, dtype=self.dtype)
            ang[:, :2] = (torch.rand((n_reset, 2), device=self.dev, dtype=self.dtype) - 0.5) \
                         * (2 * math.pi / 9)
            ang[:, 2]  = torch.rand(n_reset, device=self.dev, dtype=self.dtype) \
                         * 2 * math.pi - math.pi
            rates  = (torch.rand((n_reset, 3), device=self.dev, dtype=self.dtype) - 0.5) * 0.2
            motors = torch.rand((n_reset, self.num_motors), device=self.dev, dtype=self.dtype) * 2 - 1
        else:
            self.target_gates[mask] = 0
            pos    = self.start_pos_t.unsqueeze(0).expand(n_reset, -1).clone()
            vel    = torch.zeros((n_reset, 3), device=self.dev, dtype=self.dtype)
            ang    = torch.zeros((n_reset, 3), device=self.dev, dtype=self.dtype)
            rates  = torch.zeros((n_reset, 3), device=self.dev, dtype=self.dtype)
            motors = torch.zeros((n_reset, self.num_motors), device=self.dev, dtype=self.dtype)

        self.world_states[mask] = torch.cat([pos, vel, ang, rates, motors], dim=1)
        self.step_counts[mask]  = 0
        self.num_gates_passed[mask] = 0
        self.filtered_acts[mask]    = 0.0

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        mask = torch.ones(self.num_envs, device=self.dev, dtype=torch.bool)
        self._reset_envs(mask)
        self._update_obs()
        return self._obs_t.cpu().numpy()

    def step_async(self, actions: np.ndarray) -> None:
        self.prev_actions_t = self.actions_t.clone()
        self.actions_t = torch.as_tensor(actions, device=self.dev, dtype=self.dtype)

    def step_wait(self):
        # ---- action filter (optional IIR smoothing) --------------------
        if self.action_filter_alpha < 1.0:
            self.filtered_acts = (
                self.action_filter_alpha * self.actions_t
                + (1.0 - self.action_filter_alpha) * self.filtered_acts
            )
            act = self.filtered_acts
        else:
            act = self.actions_t

        # ---- dynamics (entirely on device) -----------------------------
        fs  = self.world_states                          # (E, 12+N)
        fsd = self._dynamics(fs.T, act.T).T              # (E, 12+N)
        new_states = fs + self.dt * fsd

        # ---- divergence check ------------------------------------------
        diverged = ~(
            torch.isfinite(new_states) & (new_states.abs() < 1e6)
        ).all(dim=1)
        new_states[diverged] = fs[diverged]

        self.step_counts += 1

        # ---- gate geometry ---------------------------------------------
        pos_old = fs[:, 0:3]
        pos_new = new_states[:, 0:3]
        tg      = self.target_gates % self.num_gates
        gpos    = self.gate_pos_t[tg]                    # (E, 3)
        gyaw    = self.gate_yaw_t[tg]                    # (E,)

        # ---- rewards ---------------------------------------------------
        d_old = (pos_old - gpos).norm(dim=1)
        d_new = (pos_new - gpos).norm(dim=1)
        rate_pen = 0.001 * new_states[:, 9:12].norm(dim=1)
        rewards  = d_old - d_new - rate_pen

        # ---- gate passing ----------------------------------------------
        nx = gyaw.cos();  ny = gyaw.sin()
        proj_old = (pos_old[:, 0] - gpos[:, 0]) * nx + (pos_old[:, 1] - gpos[:, 1]) * ny
        proj_new = (pos_new[:, 0] - gpos[:, 0]) * nx + (pos_new[:, 1] - gpos[:, 1]) * ny
        crossed  = (proj_old < 0) & (proj_new > 0)
        in_gate  = (pos_new - gpos).abs().amax(dim=1) < (self.gate_size / 2)
        gate_passed = crossed & in_gate

        rewards[gate_passed] += self.gate_reward
        final_gate = gate_passed & (self.target_gates == self.num_gates - 1)
        rewards[final_gate] += 10.0

        # ---- out of bounds ---------------------------------------------
        oob = (
            (new_states[:, 0] < self.x_bounds[0]) | (new_states[:, 0] > self.x_bounds[1]) |
            (new_states[:, 1] < self.y_bounds[0]) | (new_states[:, 1] > self.y_bounds[1]) |
            (new_states[:, 2] < self.z_bounds[0]) | (new_states[:, 2] > self.z_bounds[1])
        )
        rewards[oob | diverged] = -10.0

        # ---- episode termination ---------------------------------------
        max_steps_reached = self.step_counts >= self.max_steps
        dones = max_steps_reached | oob | diverged

        # ---- update state and gate targets -----------------------------
        self.target_gates[gate_passed] = (self.target_gates[gate_passed] + 1) % self.num_gates
        self.num_gates_passed[gate_passed] += 1
        gates_passed_snapshot = self.num_gates_passed.clone()

        self.world_states = new_states
        self._reset_envs(dones)
        self._update_obs()

        # ---- convert to numpy for SB3 ----------------------------------
        # Transfer ALL boolean tensors in one batch before building infos.
        # Never call .item() inside the per-env loop — each .item() on a CUDA
        # tensor forces a full GPU sync, creating O(num_envs) stalls per step.
        obs_np          = self._obs_t.cpu().numpy()
        rewards_np      = rewards.cpu().numpy()
        dones_np        = dones.cpu().numpy()
        gp_np           = gates_passed_snapshot.cpu().numpy()
        oob_np          = oob.cpu().numpy()
        gate_passed_np  = gate_passed.cpu().numpy()
        max_steps_np    = max_steps_reached.cpu().numpy()

        infos = [{"out_of_bounds": bool(oob_np[i]),
                  "gate_passed":   bool(gate_passed_np[i]),
                  "num_gates_passed": gp_np}
                 for i in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones_np[i]:
                infos[i]["terminal_observation"] = obs_np[i]
            if max_steps_np[i]:
                infos[i]["TimeLimit.truncated"] = True

        return obs_np, rewards_np, dones_np, infos

    # ---- required VecEnv stubs ----------------------------------------
    def close(self): pass
    def seed(self, seed=None): pass
    def get_attr(self, attr_name, indices=None): raise AttributeError()
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *args, indices=None, **kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def render(self, mode="human"): return {}
