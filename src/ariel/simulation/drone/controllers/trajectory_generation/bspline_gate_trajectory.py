# -*- coding: utf-8 -*-
"""
B-Spline Gate Trajectory Generator

This module provides automatic trajectory generation for gate-based racing
using periodic B-spline curves. Control points are initialized at gate positions
with tension-based offsets for exact interpolation, and can be further optimized
to find the fastest path through gates.

Author: Generated for AirEvolve project
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .bspline_utils import BSplineCurve, generate_knot_vector, evaluate_bspline_derivative
from ..utils.gate_configs import GateConfig


class BSplineGateTrajectory:
    """
    B-spline trajectory generator optimized for gate passing.

    This class creates a smooth periodic trajectory that passes through gates
    in a racing configuration. Gate control points are optimizable within gate
    bounds. A tension parameter controls how tightly the curve follows gate
    positions (default: exact interpolation through gates).

    Parameters can be optimized using CMA-ES or other optimization algorithms.
    """

    def __init__(self, gate_config: GateConfig,
                 degree: int = 3, gate_offset_scale: float = 1.0,
                 tension: float = 1.0):
        """
        Initialize B-spline gate trajectory with SINGLE periodic spline for C2 continuity.

        The trajectory uses ONE periodic B-spline through gate control points:
        gate[0] → gate[1] → ... → gate[N-1] → wraps back to gate[0]

        This ensures C2 continuity (smooth position, velocity, AND acceleration) throughout.

        Args:
            gate_config: Gate configuration with positions and yaws
            degree: B-spline degree (default: 3 for cubic)
            gate_offset_scale: Scale factor for gate offset bounds (default: 1.0)
                              Offsets are bounded by ±gate_size * gate_offset_scale
            tension: How tightly the B-spline follows gate positions (0.0–1.0).
                     At 0.0, control points equal gate positions (default B-spline approximation).
                     At 1.0, control points are adjusted so the curve interpolates exactly
                     through gate positions. Default: 1.0.
        """
        self.gate_config = gate_config
        self.degree = degree
        self.gate_offset_scale = gate_offset_scale
        self.tension = tension

        # Extract gate information
        self.gate_positions = np.array(gate_config.gate_pos, dtype=np.float64)
        self.gate_yaws = np.array(gate_config.gate_yaw, dtype=np.float64)
        self.gate_size = gate_config.gate_size
        self.n_gates = len(self.gate_positions)
        self.periodic = getattr(gate_config, 'periodic', True)

        # Control points: gate[0], gate[1], ..., gate[N-1] (periodic)
        self.n_total_control_points = self.n_gates

        # Default parameters (will be overridden by set_parameters)
        self.gate_offsets = None
        self.total_time = 20.0
        self.velocity_scale = 1.0
        self.startup_time = 3.0  # Time for quintic ramp from rest (seconds)

        # Initialize with default values
        self._initialize_default_parameters()

        # Single B-spline curve (periodic, includes all control points)
        self.spline = None
        self._rebuild_spline()

    def _initialize_default_parameters(self):
        """Initialize control points with default values including tension-based offsets."""
        # Gate offsets: initialize with tension-based correction
        # The offset compensates for B-spline approximation error so the curve
        # interpolates through gate positions when tension=1.0.
        # Formula: offset_i = tension * (2*G_i - G_{i-1} - G_{i+1}) / 4
        self.gate_offsets = np.zeros((self.n_gates, 3))
        for i in range(self.n_gates):
            if self.periodic:
                prev_gate = self.gate_positions[(i - 1) % self.n_gates]
                next_gate = self.gate_positions[(i + 1) % self.n_gates]
            else:
                prev_gate = self.gate_positions[max(0, i - 1)]
                next_gate = self.gate_positions[min(self.n_gates - 1, i + 1)]
            self.gate_offsets[i] = self.tension * (2 * self.gate_positions[i] - prev_gate - next_gate) / 4

    def get_all_control_points(self) -> np.ndarray:
        """
        Get all control points for the single periodic spline.

        Returns:
            Array of control points: [gate[0]+offset[0], ..., gate[N-1]+offset[N-1]]
        """
        return self.gate_positions + self.gate_offsets

    def _rebuild_spline(self):
        """Rebuild the single periodic B-spline curve."""
        all_cps = self.get_all_control_points()

        # Create spline with appropriate boundary condition
        boundary = 'periodic' if self.periodic else 'clamped'
        self.spline = BSplineCurve(all_cps, degree=self.degree, boundary=boundary)

        # Find the best starting parameter u for the trajectory (closest to desired starting position)
        self._find_optimal_start_parameter()

    def _find_optimal_start_parameter(self):
        """
        Find the parameter u on the spline that is closest to the desired starting position
        AND that leads naturally toward gate 0 (not away from it).

        This ensures that in gate-only mode, the drone starts near the configured starting position
        and moves in the correct direction through the gates.
        """
        # For non-periodic splines, start at the beginning of the curve
        if not self.periodic:
            self._u_start = self.spline.u_min
            return

        # Get the desired starting position and first gate position
        desired_start = self.get_start_position()
        first_gate_pos = self.gate_positions[0] + self.gate_offsets[0]

        # Sample the spline at many points to find candidates close to start position
        u_min = self.spline.u_min
        u_max = self.spline.u_max
        u_samples = np.linspace(u_min, u_max - 0.01, 200)  # Sample 200 points along spline

        # Find all points close to the starting position (within 2m)
        candidates = []
        for u in u_samples:
            pos = self.spline.position(u)
            distance_to_start = np.linalg.norm(pos - desired_start)
            if distance_to_start < 2.5:  # Within 2.5m of starting position
                candidates.append((u, pos, distance_to_start))

        if not candidates:
            # Fallback: just use u_min if no candidates found
            self._u_start = u_min
            return

        # Among candidates, choose the one where the trajectory moves TOWARD gate 0
        # Check which direction the spline goes from each candidate by looking ahead
        best_u = u_min
        best_score = float('inf')

        for u_candidate, pos_candidate, dist_to_start in candidates:
            # Look ahead a bit on the trajectory (0.5 parameter units)
            u_ahead = u_candidate + 0.5
            if self.periodic:
                if u_ahead >= u_max:
                    u_ahead = u_min + (u_ahead - u_min) % (u_max - u_min)
            else:
                u_ahead = min(u_ahead, u_max - 1e-10)

            pos_ahead = self.spline.position(u_ahead)

            # Calculate distance from ahead position to first gate
            dist_ahead_to_gate0 = np.linalg.norm(pos_ahead - first_gate_pos)
            dist_candidate_to_gate0 = np.linalg.norm(pos_candidate - first_gate_pos)

            # We want the trajectory to be getting CLOSER to gate 0, not farther
            # Positive value means moving toward gate, negative means moving away
            approach_score = dist_candidate_to_gate0 - dist_ahead_to_gate0

            # Score combines: close to start position AND moving toward gate 0
            # We prioritize moving toward gate 0 (higher weight)
            score = dist_to_start * 0.3 - approach_score * 2.0

            if score < best_score:
                best_score = score
                best_u = u_candidate

        # Store the optimal starting parameter
        self._u_start = best_u

    def set_parameters(self, params: np.ndarray):
        """
        Set trajectory parameters from optimization vector.

        Parameter vector structure:
        [
            # Gate position offsets (n_gates × 3)
            g0_dx, g0_dy, g0_dz, g1_dx, g1_dy, g1_dz, ...

            # Timing/velocity parameters (3)
            total_time, velocity_scale, startup_time
        ]

        Args:
            params: Parameter vector
        """
        expected_length = self.n_gates * 3 + 3

        if len(params) != expected_length:
            raise ValueError(f"Expected {expected_length} parameters, got {len(params)}")

        idx = 0

        # Extract gate offsets
        gate_offsets_flat = params[idx:idx + self.n_gates * 3]
        self.gate_offsets = gate_offsets_flat.reshape((self.n_gates, 3))
        idx += self.n_gates * 3

        # Extract timing parameters
        self.total_time = params[idx]
        self.velocity_scale = params[idx + 1]
        self.startup_time = params[idx + 2]

        # Rebuild spline with new parameters
        self._rebuild_spline()

    def get_default_parameters(self) -> np.ndarray:
        """
        Get default parameter vector for initialization.

        Returns:
            Default parameter vector
        """
        params = []

        # Gate offsets
        params.extend(self.gate_offsets.flatten())

        # Timing parameters
        params.extend([self.total_time, self.velocity_scale, self.startup_time])

        return np.array(params)

    def get_parameter_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds for optimization parameters.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = []
        upper = []

        # Bounds for gate offsets (limited by gate size)
        max_offset = self.gate_size * self.gate_offset_scale
        for _ in range(self.n_gates):
            lower.extend([-max_offset, -max_offset, -max_offset])
            upper.extend([max_offset, max_offset, max_offset])

        # Bounds for timing parameters
        lower.extend([5.0, 0.3, 1.0])     # total_time, velocity_scale, startup_time
        upper.extend([30.0, 2.0, 10.0])   # total_time, velocity_scale, startup_time

        return np.array(lower), np.array(upper)

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate trajectory at time t using single periodic spline.

        The spline is naturally C2 continuous everywhere since it's a single curve.
        Time parameterization uses quintic ramp for smooth startup from rest.

        Args:
            t: Time in seconds

        Returns:
            Tuple of (position, velocity, acceleration)
            Each is a numpy array of shape (3,)
        """
        if self.spline is None:
            raise RuntimeError("Spline not initialized. Call set_parameters first.")

        # Map time to spline parameter
        u, du_dt, d2u_dt2 = self._time_to_parameter(t)

        # Evaluate single spline (guaranteed C2 continuous!)
        position = self.spline.position(u)
        velocity = self.spline.velocity(u, du_dt)
        acceleration = self.spline.acceleration(u, du_dt, d2u_dt2)

        return position, velocity, acceleration

    def _time_to_parameter(self, t: float) -> Tuple[float, float, float]:
        """
        Map time to B-spline parameter using piecewise parameterization:
        - Startup phase (0 to startup_time): Quintic ramp from rest
        - Loop phase (startup_time to total_time): Constant speed completing full periodic cycle

        For proper looping:
        - Startup goes from optimal start point (u_start) to some point on the spline
        - Loop completes a FULL cycle: one complete traversal of the periodic spline (u_range distance)
        - At t=total_time, we return to the same u as at t=startup_time, ensuring continuity

        Args:
            t: Time in seconds

        Returns:
            Tuple of (u, du/dt, d²u/dt²)
        """
        u_min = self.spline.u_min
        u_max = self.spline.u_max
        u_range = u_max - u_min

        # Start from optimal position (closest to desired starting position)
        # This is computed in _find_optimal_start_parameter()
        u_start = getattr(self, '_u_start', u_min)

        # For a looping trajectory, the loop phase should traverse the full spline range
        # This ensures when t wraps from total_time to startup_time, position/velocity/acceleration are continuous
        loop_time = self.total_time - self.startup_time

        # The loop phase traverses a full u_range for periodic, or the remaining range for non-periodic
        loop_u_distance = u_range
        loop_speed = loop_u_distance / loop_time if loop_time > 0 else 1.0

        # For non-periodic splines, the startup already covers some parameter distance,
        # so the loop phase only needs to cover the remainder to reach u_max exactly
        if not self.periodic:
            startup_u_consumed = loop_speed * self.startup_time
            remaining_u = u_range - startup_u_consumed
            loop_u_distance = max(remaining_u, 0.0)

        if t < self.startup_time:
            # Startup phase: quintic ramp from rest at u_start
            # We want to ramp up to the loop speed smoothly
            T = self.startup_time
            t_clamped = np.clip(t, 0.0, T)

            # Quintic: u(t) = u_start + c*t³ + d*t⁴ + e*t⁵
            # Constraints:
            #   u(0) = u_start
            #   u(T) = u_start + loop_speed * T  (distance traveled during startup)
            #   du/dt(0) = 0, d²u/dt²(0) = 0 (start from rest)
            #   du/dt(T) = loop_speed, d²u/dt²(T) = 0 (match loop speed)

            # Distance traveled during startup at constant loop speed
            startup_u_distance = loop_speed * T

            # Solve quintic system for coefficients
            A = np.array([
                [T**3,    T**4,     T**5],
                [3*T**2,  4*T**3,   5*T**4],
                [6*T,     12*T**2,  20*T**3]
            ])
            b = np.array([startup_u_distance, loop_speed, 0.0])

            try:
                coeffs = np.linalg.solve(A, b)
                c, d, e = coeffs
            except np.linalg.LinAlgError:
                # Fallback: simple quintic with zero boundary conditions
                c = 10.0 * startup_u_distance / (T ** 3)
                d = -15.0 * startup_u_distance / (T ** 4)
                e = 6.0 * startup_u_distance / (T ** 5)

            u = u_start + c * t_clamped ** 3 + d * t_clamped ** 4 + e * t_clamped ** 5
            du_dt = (3.0 * c * t_clamped ** 2 + 4.0 * d * t_clamped ** 3 + 5.0 * e * t_clamped ** 4) * self.velocity_scale
            d2u_dt2 = (6.0 * c * t_clamped + 12.0 * d * t_clamped ** 2 + 20.0 * e * t_clamped ** 3) * self.velocity_scale

        else:
            # Loop phase: constant speed completing full periodic cycle
            t_loop = t - self.startup_time

            # Map loop time to spline parameter
            if loop_time > 0:
                if self.periodic:
                    # Use modulo for repeated loops (allows t > total_time)
                    t_normalized = (t_loop % loop_time) / loop_time
                else:
                    # Clamp for non-periodic: traverse once then hold at end
                    t_normalized = min(t_loop / loop_time, 1.0)

                # Start from where startup ended and traverse full u_range
                u_startup_end = u_start + loop_speed * self.startup_time
                u = u_startup_end + loop_u_distance * t_normalized
                du_dt = loop_speed * self.velocity_scale
            else:
                u = u_start
                du_dt = 0.0

            d2u_dt2 = 0.0

        # Handle parameter bounds
        if self.periodic:
            # Wrap parameter for periodic spline
            if u >= u_max:
                u = u_min + (u - u_min) % u_range
        else:
            # Clamp parameter for non-periodic spline
            u = np.clip(u, u_min, u_max - 1e-10)

        u = np.clip(u, u_min, u_max - 1e-10)

        return u, du_dt, d2u_dt2

    def sample_trajectory(self, dt: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Sample the entire trajectory at regular time intervals.

        Args:
            dt: Time step in seconds

        Returns:
            Dictionary with keys:
            - 'time': Array of time values
            - 'position': Array of positions (N, 3)
            - 'velocity': Array of velocities (N, 3)
            - 'acceleration': Array of accelerations (N, 3)
        """
        n_samples = int(self.total_time / dt) + 1
        times = np.linspace(0, self.total_time, n_samples)

        positions = []
        velocities = []
        accelerations = []

        for t in times:
            pos, vel, acc = self.evaluate(t)
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)

        return {
            'time': times,
            'position': np.array(positions),
            'velocity': np.array(velocities),
            'acceleration': np.array(accelerations)
        }

    def check_gate_proximity(self, dt: float = 0.01) -> np.ndarray:
        """
        Check minimum distance from trajectory to each gate.

        Args:
            dt: Time step for sampling trajectory

        Returns:
            Array of minimum distances to each gate
        """
        trajectory = self.sample_trajectory(dt)
        positions = trajectory['position']

        min_distances = np.full(self.n_gates, np.inf)

        for gate_idx in range(self.n_gates):
            gate_pos = self.gate_positions[gate_idx]
            distances = np.linalg.norm(positions - gate_pos, axis=1)
            min_distances[gate_idx] = np.min(distances)

        return min_distances

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return self.n_gates * 3 + 3  # gate offsets + timing

    def get_gate_offset_count(self) -> int:
        """Get number of gate offset parameters."""
        return self.n_gates * 3

    def get_timing_parameter_count(self) -> int:
        """Get number of timing parameters."""
        return 3

    def get_gate_offset_parameters(self) -> np.ndarray:
        """
        Get gate offset parameters as a flat array.

        Returns:
            Flattened array of gate offsets [g0_dx, g0_dy, g0_dz, g1_dx, ...]
        """
        return self.gate_offsets.flatten()

    def get_timing_parameters(self) -> np.ndarray:
        """
        Get timing/velocity parameters.

        Returns:
            Array of [total_time, velocity_scale, startup_time]
        """
        return np.array([self.total_time, self.velocity_scale, self.startup_time])

    def set_gate_offset_parameters(self, params: np.ndarray):
        """
        Set gate offset parameters and rebuild spline.

        Args:
            params: Flat array of gate offsets (n_gates * 3 elements)
        """
        expected_length = self.n_gates * 3
        if len(params) != expected_length:
            raise ValueError(f"Expected {expected_length} gate offset parameters, got {len(params)}")

        self.gate_offsets = params.reshape((self.n_gates, 3))
        self._rebuild_spline()

    def set_timing_parameters(self, params: np.ndarray):
        """
        Set timing/velocity parameters (no spline rebuild needed).

        Args:
            params: Array of [total_time, velocity_scale, startup_time]
        """
        if len(params) != 3:
            raise ValueError(f"Expected 3 timing parameters, got {len(params)}")

        self.total_time = params[0]
        self.velocity_scale = params[1]
        self.startup_time = params[2]

    def get_parameter_bounds_by_group(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get parameter bounds organized by group.

        Returns:
            Dictionary with keys 'gate_offsets' and 'timing', each containing
            (lower_bounds, upper_bounds) tuples
        """
        # Gate offset bounds
        max_offset = self.gate_size * self.gate_offset_scale
        gate_lower = np.full(self.n_gates * 3, -max_offset)
        gate_upper = np.full(self.n_gates * 3, max_offset)

        # Timing parameter bounds
        timing_lower = np.array([5.0, 0.3, 1.0])     # total_time, velocity_scale, startup_time
        timing_upper = np.array([30.0, 2.0, 10.0])   # total_time, velocity_scale, startup_time

        return {
            'gate_offsets': (gate_lower, gate_upper),
            'timing': (timing_lower, timing_upper)
        }

    def get_start_position(self) -> np.ndarray:
        """
        Get the starting position for the trajectory.

        Always uses starting_pos from gate configuration if available,
        otherwise calculates position behind gate 0.

        Note: This returns the physical starting position for the drone,
        NOT necessarily a point on the B-spline curve. In gate-only mode,
        the drone starts here but the trajectory is a periodic loop through gates.

        Returns:
            Starting position [x, y, z]
        """
        # Use starting position from gate config if available (preferred)
        if hasattr(self.gate_config, 'starting_pos') and self.gate_config.starting_pos is not None:
            return np.array(self.gate_config.starting_pos, dtype=np.float64)

        # Fallback: calculate start position behind gate 0 by 2 meters
        first_gate = self.gate_positions[0]
        first_gate_yaw = self.gate_yaws[0]

        distance_behind = 2.0  # meters
        start_pos = first_gate - distance_behind * np.array([
            np.cos(first_gate_yaw),
            np.sin(first_gate_yaw),
            0.0
        ])

        return start_pos

    def get_info(self) -> Dict[str, Any]:
        """
        Get trajectory information.

        Returns:
            Dictionary with trajectory configuration
        """
        return {
            'n_gates': self.n_gates,
            'n_total_control_points': self.n_total_control_points,
            'degree': self.degree,
            'tension': self.tension,
            'total_time': self.total_time,
            'velocity_scale': self.velocity_scale,
            'startup_time': self.startup_time,
            'n_parameters': self.get_parameter_count(),
            'gate_size': self.gate_size,
            'gate_offset_scale': self.gate_offset_scale,
            'spline_type': 'single_periodic' if self.periodic else 'single_clamped'
        }


def create_gate_trajectory_from_config(gate_config_name: str) -> BSplineGateTrajectory:
    """
    Convenience function to create trajectory from gate configuration name.

    Args:
        gate_config_name: Name of gate configuration ('figure8', 'circle', etc.)

    Returns:
        BSplineGateTrajectory instance

    Example:
        >>> from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
        >>> traj = create_gate_trajectory_from_config('figure8')
    """
    from ..utils.gate_configs import GATE_CONFIGS

    if gate_config_name not in GATE_CONFIGS:
        raise ValueError(f"Unknown gate configuration: {gate_config_name}")

    gate_config = GATE_CONFIGS[gate_config_name]
    return BSplineGateTrajectory(gate_config)
