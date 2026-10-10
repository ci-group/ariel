"""
Drone Simulator with Propeller-Based Configuration

Reference-form symbolic dynamics. Mirrors
`optimal_quad_control_RL/quad_race_env.py:22-103` (the sysid'd 5-inch
canonical model) but parameterized by airevolve's morphology.

Per-step parity against the reference is verified at machine precision
(<1e-9 rel err) by `unit_tests/test_dynamics_parity.py`.

See `experimentation/RUNTIME_DYNAMICS_MIGRATION.md` (the migration plan)
and `experimentation/RL_TRAINING_FIXES.md` (the diagnostic chain that
justified the rewrite).
"""

import warnings
import numpy as np
from sympy import Array, Matrix, cos, lambdify, sin, sqrt, symbols, tan
from .drone_configuration import DroneConfiguration
from .propeller_data import create_standard_propeller_config
from .dynamics_params import derive_reference_params, W_MAX_N, W_MIN_N


def _invert_sqrt_poly(Wc, w_max, w_min, k):
    """Solve `Wc = (w_max - w_min)·sqrt(k·U² + (1-k)·U) + w_min` for U∈[0,1].

    Used by `update_from_controller` to convert a controller's commanded
    steady-state motor speed (rad/s) into the env action.
    """
    Wc = float(np.clip(Wc, w_min, w_max))
    z = (Wc - w_min) / (w_max - w_min)
    z2 = z * z
    if abs(k) < 1e-12:
        return z2
    disc = (1.0 - k) ** 2 + 4.0 * k * z2
    return (-(1.0 - k) + np.sqrt(disc)) / (2.0 * k)

class DroneSimulator:
    """
    Core drone simulation framework using reference-form sysid'd dynamics.

    State vector (length 12 + N where N = num_motors):
        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w_1..w_N]
    where w_i ∈ [-1, 1] is the normalized motor speed (W_i = (w_i+1)/2 ·
    W_MAX_N + W_MIN_N rad/s; W_MAX_N=3000, W_MIN_N=0).

    Action vector (length N): u_i ∈ [-1, 1].

    `dynamics_func(full_state, action) → full_state_dot` is lambdified;
    motor model (sqrt-poly mapping + first-order lag) is baked in.

    The reference's `Bf, Bm` allocation matrices are kept as decorative
    attributes for API compatibility with consumers that read them; the
    new dynamics path uses per-motor coefficients in `self.params` instead.
    """

    def __init__(self, propellers=None, mountpoints=None, dt=0.005, gravity=9.81):
        """
        Initialize drone simulator from propeller configuration.

        Args:
            propellers (list): List of propeller dictionaries, each containing:
                - "loc": [x, y, z] position in body frame (meters)
                - "dir": [x, y, z, rotation] thrust direction and spin direction
                - "propsize": propeller size in inches (2-8) or "matched"
            mountpoints (list, optional): Accepted for API compat;
                DroneConfiguration does not currently consume it.
            dt (float): Integration time step
            gravity (float): Gravitational acceleration

        Example:
            # Standard quadrotor
            propellers = [
                {"loc": [0.11, 0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [-0.11, 0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
                {"loc": [-0.11, -0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [0.11, -0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5}
            ]
            drone = DroneSimulator(propellers=propellers)
        """

        if propellers is None:
            propellers = create_standard_propeller_config("quad", arm_length=0.11, prop_size=2)

        self.config = DroneConfiguration(propellers)

        # Decorative — kept for API compatibility with consumers that read
        # them (e.g., scripts using get_params for the legacy controller).
        # The dynamics_func uses `self.params` (per-motor coefficients) instead.
        self.Bf, self.Bm = self.config.get_allocation_matrices()
        self.num_motors = self.config.num_motors
        self.mass = self.config.mass
        self.inertia = self.config.inertia_matrix
        self.center_of_gravity = self.config.cg

        # All propellers are assumed to share the same prop_size (current
        # airevolve convention). For asymmetric prop sizes, derive_reference_params
        # would need per-motor extended params.
        prop_size = propellers[0].get("propsize", 2)
        self.prop_size = prop_size

        self.params = derive_reference_params(
            propellers=propellers,
            mass=float(self.mass),
            inertia=np.asarray(self.inertia),
            prop_size=prop_size,
            gravity=gravity,
        )

        self.dt = dt
        self.g = gravity

        self._setup_dynamics()

        # Full state: [base 12 dims, motor speeds w_1..w_N], all w in [-1, 1].
        # The mid-range value (w=0) corresponds to W = W_MAX_N/2 = 1500 rad/s,
        # which is *not* a meaningful "stopped" value. Initialize at the
        # idle floor instead: W = w_min, i.e., w_init = 2·w_min/W_MAX_N - 1.
        self._w_init = 2.0 * self.params["w_min"] / (W_MAX_N - W_MIN_N) - 1.0
        self.state = np.zeros(12 + self.num_motors, dtype=np.float64)
        self.state[12:12 + self.num_motors] = self._w_init
        # Action: u_i in [-1, 1].
        self.actions = np.zeros(self.num_motors, dtype=np.float64)
        # Legacy alias kept for API compatibility (interpreted as action).
        self.motor_commands = self.actions

        self.time_history = []
        self.state_history = []
        self.control_history = []
    
    @classmethod
    def create_standard_drone(cls, drone_type="quad", arm_length=0.11, prop_size=2, **kwargs):
        """
        Create standard drone configuration.
        
        Args:
            drone_type (str): Type of drone ('quad', 'hex', 'tri', 'octo')
            arm_length (float): Length of drone arms in meters
            prop_size (int): Propeller size in inches (4-8)
            **kwargs: Additional arguments for DroneSimulator
            
        Returns:
            DroneSimulator: Configured drone simulator
        """
        propellers = create_standard_propeller_config(drone_type, arm_length, prop_size)
        return cls(propellers=propellers, **kwargs)
    
    def _setup_dynamics(self):
        """Build the lambdified reference-form dynamics function.

        Mirrors `experimentation/reference_drone_sim.py:_build_dynamics_func`
        but generalized to N motors via `self.params` (per-motor signed
        coefficients computed by `derive_reference_params`). Per-step parity
        against the reference for the canonical 4-motor 2-inch quad is
        verified by `unit_tests/test_dynamics_parity.py`.

        The lambdified function has signature
        `(full_state[12+N], action[N]) → full_state_dot[12+N]`. Motor model
        (sqrt-poly mapping U → Wc, then first-order lag dW = (Wc-W)/tau) is
        baked in. F and M are treated as accelerations directly — mass and
        inertia are absorbed into the per-motor coefficients.
        """
        n = self.num_motors

        # Base state symbols (12 dims).
        base_state_syms = symbols('x y z v_x v_y v_z phi theta psi p q r')
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = base_state_syms
        # Motor state symbols (N dims).
        motor_state_syms = list(symbols(f'w_1:{n + 1}'))
        full_state_syms = list(base_state_syms) + motor_state_syms
        # Control (action) symbols (N dims).
        control_syms = list(symbols(f'u_1:{n + 1}'))

        # Rotation matrix (body → world).
        Rx = Matrix([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
        Ry = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
        Rz = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
        R = Rz * Ry * Rx

        # Body-frame velocity (drag computed in body frame).
        vbx, vby, _vbz = R.T @ Matrix([vx, vy, vz])

        # Unnormalize motor speed: w in [-1, 1] → W in [W_MIN_N, W_MAX_N] rad/s.
        # W_MAX_N=3000 is independent of the physical w_max — intentional in the
        # reference; state may exceed [-1, 1] under hard throttle (transient).
        W = [(w_i + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N for w_i in motor_state_syms]
        # Action: u in [-1, 1] → U in [0, 1].
        U = [(u_i + 1) / 2 for u_i in control_syms]

        p_dict = self.params
        k_w = p_dict["k_w"]
        k_x = p_dict["k_x"]
        k_y = p_dict["k_y"]
        k_p_signed = p_dict["k_p_signed"]
        k_q_signed = p_dict["k_q_signed"]
        k_r_signed = p_dict["k_r_signed"]
        k_r_react_signed = p_dict["k_r_react_signed"]
        tau = p_dict["tau"]
        k = p_dict["k"]
        w_min = p_dict["w_min"]
        w_max = p_dict["w_max"]

        # Motor command: action U → target speed Wc via sqrt-polynomial.
        Wc = [(w_max - w_min) * sqrt(k * U_i**2 + (1 - k) * U_i) + w_min for U_i in U]
        # First-order lag.
        d_W = [(Wc_i - W_i) / tau for Wc_i, W_i in zip(Wc, W)]
        # Convert dW (rad/s²) back to normalized derivative (1/s).
        d_w = [d_W_i / (W_MAX_N - W_MIN_N) * 2 for d_W_i in d_W]

        # Forces (treated as accelerations directly — mass implicit in k_w).
        sum_W = sum(W)
        sum_W2 = sum(W_i**2 for W_i in W)
        T = -k_w * sum_W2
        Dx = -k_x * vbx * sum_W
        Dy = -k_y * vby * sum_W

        # Moments (treated as angular accelerations directly — inertia
        # absorbed into k_p/k_q/k_r). Signs are baked into the per-motor
        # coefficients by derive_reference_params (so the symbolic build
        # is morphology-agnostic).
        Mx = sum(k_p_signed[i] * W[i]**2 for i in range(n))
        My = sum(k_q_signed[i] * W[i]**2 for i in range(n))
        Mz = (
            sum(k_r_signed[i] * W[i] for i in range(n))
            + sum(k_r_react_signed[i] * d_W[i] for i in range(n))
        )

        # Translational kinematics.
        d_x = vx
        d_y = vy
        d_z = vz

        # Translational dynamics: gravity + body-frame forces rotated to world.
        accel = Matrix([0, 0, self.g]) + R @ Matrix([Dx, Dy, T])
        d_vx, d_vy, d_vz = accel

        # Euler-angle kinematics (singular at theta=±π/2; reset envs guard against this).
        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        # Rotational dynamics — direct, no inertia inversion.
        d_p = Mx
        d_q = My
        d_r = Mz

        state_dot = [
            d_x, d_y, d_z,
            d_vx, d_vy, d_vz,
            d_phi, d_theta, d_psi,
            d_p, d_q, d_r,
        ] + d_w  # length 12 + n

        self.dynamics_func = lambdify(
            (Array(full_state_syms), Array(control_syms)),
            Array(state_dot),
            'numpy',
        )

    def get_configuration_info(self):
        """
        Get comprehensive drone configuration information.
        
        Returns:
            dict: Complete configuration including physical properties and capabilities
        """
        info = self.config.get_physical_properties()
        info.update(self.config.get_motor_configuration_info())
        info['propeller_configuration'] = self.config.propellers
        return info
    
    def get_propeller_info(self):
        """
        Get propeller configuration details.
        
        Returns:
            list: List of propeller specifications
        """
        return [prop.copy() for prop in self.config.propellers]
    
    def set_state(self, position=None, velocity=None, attitude=None, angular_velocity=None):
        """
        Set drone state.
        
        Args:
            position: [x, y, z] in world frame
            velocity: [vx, vy, vz] in world frame  
            attitude: [phi, theta, psi] (roll, pitch, yaw) in radians
            angular_velocity: [p, q, r] in body frame
        """
        if position is not None:
            self.state[0:3] = position
        if velocity is not None:
            self.state[3:6] = velocity
        if attitude is not None:
            self.state[6:9] = attitude
        if angular_velocity is not None:
            self.state[9:12] = angular_velocity
    
    def get_state(self):
        """Get current state as dictionary."""
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(), 
            'attitude': self.state[6:9].copy(),
            'angular_velocity': self.state[9:12].copy(),
            'time': len(self.time_history) * self.dt
        }
    
    def set_motor_commands(self, commands):
        """
        Set actions (motor inputs in [-1, 1]).

        Args:
            commands: Array of actions in range [-1, 1]. Legacy callers
                passing [0, 1] will get clipped to [-1, 1] but should migrate
                to the new range — the env layer (DroneGateEnv) maps actions
                in [-1, 1] directly into dynamics_func.
        """
        self.actions = np.clip(commands, -1, 1)
        self.motor_commands = self.actions

    def step(self, motor_commands=None):
        """
        Advance simulation by one time step using RK4 integration on the
        full 12+N state. The dynamics_func handles the motor model
        internally; pass actions in [-1, 1].

        Args:
            motor_commands: Optional actions for this step (in [-1, 1]).
        """
        if motor_commands is not None:
            self.set_motor_commands(motor_commands)

        with np.errstate(all='ignore'):
            k1 = self.dt * np.asarray(self.dynamics_func(self.state, self.actions))
            k2 = self.dt * np.asarray(self.dynamics_func(self.state + 0.5 * k1, self.actions))
            k3 = self.dt * np.asarray(self.dynamics_func(self.state + 0.5 * k2, self.actions))
            k4 = self.dt * np.asarray(self.dynamics_func(self.state + k3, self.actions))
            self.state = self.state + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # Detect numerical divergence (e.g. Euler angle singularity at ±90° pitch)
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)):
            raise RuntimeError("Numerical divergence in drone state (likely Euler angle singularity)")

        self.time_history.append(len(self.time_history) * self.dt)
        self.state_history.append(self.state.copy())
        self.control_history.append(self.actions.copy())

        return self.get_state()
    
    def simulate(self, time_span, control_function=None):
        """
        Run simulation for specified time span.
        
        Args:
            time_span: Total simulation time
            control_function: Function that takes (time, state) and returns motor commands
        """
        num_steps = int(time_span / self.dt)
        
        for i in range(num_steps):
            current_time = i * self.dt
            current_state = self.get_state()
            
            if control_function is not None:
                commands = control_function(current_time, current_state)
                self.step(commands)
            else:
                self.step()
    
    def reset(self):
        """Reset simulation to initial conditions."""
        self.state = np.zeros(12 + self.num_motors)
        self.state[12:12 + self.num_motors] = self._w_init
        self.actions = np.zeros(self.num_motors)
        self.motor_commands = self.actions
        self.time_history = []
        self.state_history = []
        self.control_history = []

    def _get_actual_motor_speeds(self):
        """Convert normalized motor speeds (in self.state[12:]) to rad/s."""
        motor_speeds = np.zeros(max(4, self.num_motors))
        if len(self.state) >= 12 + self.num_motors:
            w_norm = self.state[12:12 + self.num_motors]
            for i in range(self.num_motors):
                # W = (w + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
                motor_speeds[i] = (w_norm[i] + 1) / 2 * (W_MAX_N - W_MIN_N) + W_MIN_N
        return motor_speeds[:4]

    def get_params(self):
        """Get parameters in format compatible with existing controller framework."""
        Bm_corrected = self.Bm.copy()
        A_control = np.vstack([-self.Bf[2:3, :], Bm_corrected])
        mixer_fm = A_control
        first_prop = self.config.propellers[0]
        k_f, k_m = first_prop["constants"]
        w_max = first_prop["wmax"]
        hover_thrust_per_motor = (self.mass * self.g) / self.num_motors
        w_hover = np.sqrt(hover_thrust_per_motor / k_f)

        return {
            "mB": self.mass, "g": self.g, "IB": self.inertia, "invI": np.linalg.inv(self.inertia),
            "dxm": np.mean([abs(p["loc"][0]) for p in self.config.propellers if p["loc"][0] != 0]),
            "dym": np.mean([abs(p["loc"][1]) for p in self.config.propellers if p["loc"][1] != 0]),
            "dzm": 0.05, "kTh": k_f, "kTo": k_m, "w_hover": w_hover, "thr_hover": hover_thrust_per_motor,
            "mixerFM": mixer_fm, "mixerFMinv": np.linalg.pinv(mixer_fm),
            "minThr": 0.1 * self.num_motors, "maxThr": k_f * w_max**2 * self.num_motors,
            "minWmotor": 75, "maxWmotor": w_max,
            # Read tau from the reference-form params (set per prop in propeller_data.py).
            # The pre-migration value was hardcoded 0.015s; reference-form is 0.04s.
            "tau": float(self.params["tau"]), "kp": 1.0, "damp": 1.0,
            "motorc1": 8.49, "motorc0": 74.7, "motordeadband": 1, "Cd": 0.1, "IRzz": 2.7e-5,
            "useIntergral": False, "FF": (w_hover - 74.7) / 8.49
        }

    def get_drone_state(self):
        """Get state in format compatible with existing controller interfaces."""
        phi, theta, psi = self.state[6:9]
        cy, sy = np.cos(psi * 0.5), np.sin(psi * 0.5)
        cp, sp = np.cos(theta * 0.5), np.sin(theta * 0.5)
        cr, sr = np.cos(phi * 0.5), np.sin(phi * 0.5)
        quat = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

        actual_speeds = self._get_actual_motor_speeds()
        extended_state = np.zeros(21)
        extended_state[0:3], extended_state[3:7] = self.state[0:3], quat
        extended_state[7:10], extended_state[10:13] = self.state[3:6], self.state[9:12]
        for i in range(min(4, self.num_motors)):
            extended_state[13 + i*2] = actual_speeds[i]

        return {
            'state': extended_state, 'pos': self.state[0:3], 'vel': self.state[3:6], 'quat': quat,
            'omega': self.state[9:12], 'euler': np.array([0, 0, 0]), 'wMotor': actual_speeds,
            'vel_dot': np.zeros(3), 'omega_dot': np.zeros(3), 'acc': np.zeros(3),
            'thr': self.actions[:4] if self.num_motors >= 4 else np.pad(self.actions, (0, 4-self.num_motors)),
            'tor': self.actions[:4] if self.num_motors >= 4 else np.pad(self.actions, (0, 4-self.num_motors)),
            'dcm': np.eye(3)
        }

    def update_from_controller(self, t, Ts, w_cmd, wind=None):
        """Update simulation using commands from controller framework.

        Maps the controller's commanded steady-state motor speeds (rad/s)
        into the action range [-1, 1] by inverting the reference-form
        sqrt-poly motor model

            Wc = (w_max - w_min) · sqrt(k · U² + (1 - k) · U) + w_min

        for U ∈ [0, 1], then `action = 2 · U - 1`. Without this inverse,
        a hover-throttle w_cmd ≈ W_hover under-commands by ~2× because of
        the sqrt curvature and non-zero w_min idle floor.
        """
        full_w_cmd = self.w_cmd_full if hasattr(self, 'w_cmd_full') and self.w_cmd_full is not None else w_cmd
        w_min = self.params["w_min"]
        k = self.params["k"]
        propeller_w_max = [prop["wmax"] for prop in self.config.propellers]
        actions = np.zeros(self.num_motors)
        for i in range(min(len(full_w_cmd), self.num_motors)):
            w_max = propeller_w_max[i] if i < len(propeller_w_max) else propeller_w_max[0]
            actions[i] = 2.0 * _invert_sqrt_poly(full_w_cmd[i], w_max, w_min, k) - 1.0
        self.step(actions)
        return t + Ts


# Factory functions for easy drone creation
def create_quadrotor(arm_length=0.11, prop_size=2, **kwargs):
    """Create standard quadrotor configuration."""
    return DroneSimulator.create_standard_drone("quad", arm_length, prop_size, **kwargs)

def create_hexarotor(arm_length=0.10, prop_size=4, **kwargs):
    """Create standard hexarotor configuration.""" 
    return DroneSimulator.create_standard_drone("hex", arm_length, prop_size, **kwargs)

def create_tricopter(arm_length=0.12, prop_size=6, **kwargs):
    """Create standard tricopter configuration."""
    return DroneSimulator.create_standard_drone("tri", arm_length, prop_size, **kwargs)

def create_octorotor(arm_length=0.09, prop_size=4, **kwargs):
    """Create standard octorotor configuration."""
    return DroneSimulator.create_standard_drone("octo", arm_length, prop_size, **kwargs)
