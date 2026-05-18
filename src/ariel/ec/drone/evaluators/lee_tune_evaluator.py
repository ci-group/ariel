"""
Lee Controller Tuning Evaluator for Evolutionary Algorithm

This evaluator uses a 2-stage CMA-ES pipeline to optimize controller parameters
for each evolved morphology. Each individual's fitness is determined by the maximum
number of gates passed during controller tuning.

If controller tuning fails to pass any gates, the individual receives fitness of 0.

Design:
- Takes an evolved drone morphology
- Converts it to DroneInterface
- Stage 1: Optimise gains + timing (7 params: pos_P, vel_P, att_P, rate_P,
  total_time, velocity_scale, startup_time) with fixed default trajectory
- Stage 2: Optimise gains + timing + gate offset control points
  (7 + n_gates*3 params) to refine the trajectory shape
- Returns max gates passed as fitness

This integrates the tune_lee_controller_gates.py pipeline into the evolutionary loop.
"""

import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing

# Import drone simulation components
from ariel.simulation.drone import DroneInterface
from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory
from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl
from ariel.simulation.drone.controllers.utils.wind_model import Wind
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.ec.drone.inspection.utils import convert_to_cartesian, ENU_to_NED
from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import orientation_to_unit_vector

# Try to import CMA-ES
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("WARNING: 'cma' package not found. Install with: pip install cma")


# ============================================================================
# GATE CHECKING LOGIC (from tune_lee_controller_gates.py)
# ============================================================================

class GateChecker:
    """Handles gate passing detection"""

    def __init__(self, gate_pos, gate_yaw, gate_size=1.0):
        """
        Initialize gate checker

        Args:
            gate_pos: Array of gate positions [N, 3]
            gate_yaw: Array of gate yaw angles [N]
            gate_size: Size of gates in meters
        """
        self.gate_pos = gate_pos
        self.gate_yaw = gate_yaw
        self.gate_size = gate_size
        self.num_gates = len(gate_pos)
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.last_pos = None
        self.max_gate_distance = self._calculate_max_gate_distance()

    def reset(self):
        """Reset gate checker state"""
        self.current_gate_idx = 0
        self.gates_passed = 0
        self.last_pos = None

    def check_gate_passing(self, pos):
        """
        Check if drone passed through current gate

        Args:
            pos: Current position [x, y, z]

        Returns:
            True if gate was passed
        """
        if self.last_pos is None:
            self.last_pos = pos.copy()
            return False

        # Get current gate
        gate_idx = self.current_gate_idx % self.num_gates
        gate_pos = self.gate_pos[gate_idx]
        gate_yaw = self.gate_yaw[gate_idx]

        # Gate normal vector (direction perpendicular to gate plane)
        normal = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])

        # Project positions onto normal direction
        pos_old_proj = np.dot(self.last_pos - gate_pos, normal)
        pos_new_proj = np.dot(pos - gate_pos, normal)

        # Check if crossed gate plane
        crossed_plane = ((pos_old_proj < 0) and (pos_new_proj > 0)) or \
                       ((pos_old_proj > 0) and (pos_new_proj < 0))

        if crossed_plane:
            # Find intersection point
            t = -pos_old_proj / (pos_new_proj - pos_old_proj)
            intersection = self.last_pos + t * (pos - self.last_pos)

            # Transform to gate's local frame
            rel_pos = intersection - gate_pos

            # Create gate's local coordinate system
            if abs(normal[2]) < 0.9:
                up = np.array([0.0, 0.0, 1.0])
            else:
                up = np.array([0.0, 1.0, 0.0])

            right = np.cross(normal, up)
            right = right / np.linalg.norm(right)
            actual_up = np.cross(right, normal)

            # Project onto gate plane
            lateral = np.dot(rel_pos, right)
            vertical = np.dot(rel_pos, actual_up)

            # Check if within gate opening
            half_size = self.gate_size / 2.0
            within_bounds = (abs(lateral) <= half_size) and (abs(vertical) <= half_size)

            if within_bounds:
                self.gates_passed += 1
                self.current_gate_idx += 1
                self.last_pos = pos.copy()
                return True

        self.last_pos = pos.copy()
        return False

    def _calculate_max_gate_distance(self):
        """
        Calculate the maximum distance between consecutive gates in the sequence

        Returns:
            Maximum distance between any two consecutive gates
        """
        max_dist = 0.0
        for i in range(self.num_gates):
            next_idx = (i + 1) % self.num_gates
            dist = np.linalg.norm(self.gate_pos[next_idx] - self.gate_pos[i])
            max_dist = max(max_dist, dist)
        return max_dist

    def get_normalized_distance_to_next_gate(self, pos):
        """
        Calculate normalized distance to the next gate (current target)

        Args:
            pos: Current position [x, y, z]

        Returns:
            Normalized distance bonus in [0, 1], where:
            - 1.0 means at the gate position
            - 0.0 means at or beyond max_gate_distance away
            - Values capped at 0.0 if distance > max_gate_distance
        """
        # Get the next gate to pass
        gate_idx = self.current_gate_idx % self.num_gates
        gate_pos = self.gate_pos[gate_idx]

        # Calculate distance to next gate
        distance = np.linalg.norm(pos - gate_pos)

        # Normalize: closer = higher score, capped at 0 if too far
        if distance >= self.max_gate_distance:
            return 0.0

        normalized = 1.0 - (distance / self.max_gate_distance)
        return max(0.0, min(1.0, normalized))


# ============================================================================
# SIMULATION (adapted from tune_lee_controller_gates.py)
# ============================================================================

def simulate_with_gains(individual, pos_gain, vel_gain, att_gain, rate_gain,
                        gate_config, sim_time=20.0, dt=0.005,
                        bspline_timing=None, gate_offsets=None,
                        verbose=False, record_trajectory=False):
    """
    Run simulation with Lee controller for a given morphology and gains

    Args:
        individual: Evolved drone morphology (genome array)
        pos_gain, vel_gain, att_gain, rate_gain: Controller gains
        gate_config: Gate configuration class
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        bspline_timing: Optional array of [total_time, velocity_scale, startup_time].
                        If None, uses BSplineGateTrajectory defaults (20.0, 1.0, 3.0).
        gate_offsets: Optional flat array of gate offset parameters (n_gates * 3 elements).
                      If None, uses default zero offsets.
        verbose: If True, show debug output
        record_trajectory: If True, record full trajectory data at each timestep

    Returns:
        Dictionary with gates_passed, crashed, completed, flight_time.
        If record_trajectory=True, also includes 'trajectory' dict with numpy arrays:
            positions, velocities, euler_angles, angular_velocities, motor_commands, gate_passes
    """
    try:
        # Convert genome to DroneInterface
        # Individual format: N x 6 array [r, theta, phi, pitch, yaw, direction]
        propellers = []
        for arm in individual:
            r, theta, phi, motor_pitch, motor_yaw, direction = arm

            # Position: genome spherical (elevation convention, ENU) -> Cartesian NED
            ex, ey, ez = convert_to_cartesian(r, theta, phi)
            x, y, z = ENU_to_NED(ex, ey, ez)

            # Motor rotation direction
            rot = "ccw" if direction < 0.5 else "cw"

            # Motor direction: orientation_to_unit_vector returns the motor force
            # direction in NED. Use directly — same convention as the hover check
            # (hovering_info.get_sim) which validates that the drone can hover.
            # Do NOT negate: negation inverts force directions, making hoverable
            # drones produce downward force in the simulation.
            motor_dir = orientation_to_unit_vector(0.0, motor_pitch, motor_yaw)

            propellers.append({
                "loc": [float(x), float(y), float(z)],
                "dir": [float(motor_dir[0]), float(motor_dir[1]), float(motor_dir[2]), rot],
                "propsize": 2
            })

        quad = DroneInterface(0, propellers=propellers)

        # Create Lee controller with specified gains
        lee_gains = {
            'pos_P_gain': np.array([pos_gain] * 3),
            'vel_P_gain': np.array([vel_gain] * 3),
            'att_P_gain': np.array([att_gain] * 3),
            'rate_P_gain': np.array([rate_gain] * 3)
        }
        ctrl = LeeGeometricControl(quad, yawType=1, orient='NED',
                                   auto_scale_gains=False, **lee_gains)

        # Create B-spline trajectory
        bspline_traj = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
        bspline_params = bspline_traj.get_default_parameters()
        bspline_traj.set_parameters(bspline_params)

        # Override timing if provided
        if bspline_timing is not None:
            bspline_traj.set_timing_parameters(np.asarray(bspline_timing))

        # Override gate offsets if provided
        if gate_offsets is not None:
            bspline_traj.set_gate_offset_parameters(np.asarray(gate_offsets))

        # CRITICAL: Set drone initial state to match trajectory at t=0
        # This eliminates startup transients from position/velocity/yaw errors
        start_pos, _, _ = bspline_traj.evaluate(0.0)

        # Compute initial yaw from trajectory's heading direction at t=0.05s
        # (t=0 has zero velocity due to quintic startup ramp)
        _, vel_050, _ = bspline_traj.evaluate(0.05)
        if np.linalg.norm(vel_050[:2]) > 0.001:  # If horizontal velocity > 0.001 m/s
            initial_yaw = np.arctan2(vel_050[1], vel_050[0])
        else:
            # Fallback to first gate direction if velocity is too small
            initial_yaw = gate_config.gate_yaw[0]

        initial_euler = np.array([0.0, 0.0, initial_yaw])

        # Set drone state: position from t=0, zero velocity (trajectory starts from rest),
        # yaw from trajectory heading to prevent yaw error torque spike
        quad.drone_sim.set_state(position=start_pos, velocity=np.zeros(3),
                                attitude=initial_euler, angular_velocity=np.zeros(3))

        # Create Trajectory wrapper (xyzType=15 for B-spline)
        from ariel.simulation.drone.controllers.trajectory_generation.trajectory import Trajectory
        traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                         gate_config=gate_config)
        traj.bspline_trajectory = bspline_traj

        # Create wind model (no wind)
        wind = Wind('None', 2.0, 90, -15)

        # Initialize gate checker
        gate_checker = GateChecker(gate_config.gate_pos, gate_config.gate_yaw,
                                  gate_config.gate_size)

        # Get initial desired state and command
        sDes = traj.desiredState(0, dt, quad)
        ctrl.controller(sDes, quad, "xyz_pos", dt)

        # Run simulation
        t = 0
        crashed = False
        num_steps = int(sim_time / dt)

        # Trajectory recording
        if record_trajectory:
            traj_positions = []
            traj_velocities = []
            traj_euler = []
            traj_omega = []
            traj_w_cmd = []
            traj_gate_passes = []

        for step in range(num_steps):
            # Update dynamics
            try:
                quad.update(t, dt, ctrl.w_cmd, wind)
            except RuntimeError:
                crashed = True
                break
            t += dt

            # Get desired state
            sDes = traj.desiredState(t, dt, quad)

            # Generate control commands
            ctrl.controller(sDes, quad, "xyz_pos", dt)

            # Check gate passing
            gate_passed = gate_checker.check_gate_passing(quad.pos)

            # Record trajectory data
            if record_trajectory:
                traj_positions.append(quad.pos.copy())
                traj_velocities.append(quad.vel.copy())
                traj_euler.append(quad.euler.copy())
                traj_omega.append(quad.omega.copy())
                traj_w_cmd.append(ctrl.w_cmd.copy())
                traj_gate_passes.append(gate_passed)

            # Check bounds
            if (quad.pos[0] < gate_config.x_bounds[0] or
                quad.pos[0] > gate_config.x_bounds[1] or
                quad.pos[1] < gate_config.y_bounds[0] or
                quad.pos[1] > gate_config.y_bounds[1] or
                quad.pos[2] < gate_config.z_bounds[0] or
                quad.pos[2] > gate_config.z_bounds[1]):
                crashed = True
                break

            # Check for excessive error (instability)
            pos_error = np.linalg.norm(quad.pos - sDes[0:3])
            if pos_error > 10.0:
                crashed = True
                break

        # Calculate normalized distance to next gate at end of simulation
        distance_bonus = gate_checker.get_normalized_distance_to_next_gate(quad.pos)

        result = {
            'gates_passed': gate_checker.gates_passed,
            'distance_bonus': distance_bonus,
            'crashed': crashed,
            'completed': not crashed,
            'flight_time': t,
            'success': True
        }

        if record_trajectory:
            result['trajectory'] = {
                'positions': np.array(traj_positions),
                'velocities': np.array(traj_velocities),
                'euler_angles': np.array(traj_euler),
                'angular_velocities': np.array(traj_omega),
                'motor_commands': np.array(traj_w_cmd),
                'gate_passes': np.array(traj_gate_passes),
            }

        return result

    except Exception as e:
        if verbose:
            print(f"Simulation terminated early: {e}")
        return {
            'gates_passed': 0,
            'distance_bonus': 0.0,
            'crashed': True,
            'completed': False,
            'flight_time': 0,
            'success': False,
            'error': str(e)
        }


# ============================================================================
# MULTIPROCESSING WRAPPER
# ============================================================================

def _evaluate_solution_wrapper(args):
    """Wrapper function for parallel evaluation.

    Supports variable-length solution vectors:
    - 4 params: gains only (pos_P, vel_P, att_P, rate_P)
    - 7 params: gains + timing (+ total_time, velocity_scale, startup_time)
    - 7+N params: gains + timing + gate offsets (N = n_gates * 3)
    """
    (params, individual, gate_config, sim_time, dt, bspline_timing) = args

    pos_g, vel_g, att_g, rate_g = params[0:4]

    # If solution vector includes timing params, use them
    if len(params) >= 7:
        bspline_timing = params[4:7]

    # If solution vector includes gate offsets, extract them
    gate_offsets = None
    if len(params) > 7:
        gate_offsets = params[7:]

    # Run simulation
    result = simulate_with_gains(
        individual, pos_g, vel_g, att_g, rate_g,
        gate_config, sim_time, dt,
        bspline_timing=bspline_timing,
        gate_offsets=gate_offsets,
        verbose=False
    )

    if result['success']:
        gates = result['gates_passed']
        distance_bonus = result['distance_bonus']
        penalty = 100 if result['crashed'] else 0

        # Fitness includes gates passed + normalized distance to next gate
        fitness = gates + distance_bonus
        score = -fitness + penalty

        result['score'] = score
        result['fitness'] = fitness
        result['gains'] = {
            'pos_P': pos_g,
            'vel_P': vel_g,
            'att_P': att_g,
            'rate_P': rate_g
        }
        if len(params) >= 7:
            result['bspline_timing'] = list(params[4:7])
        if gate_offsets is not None:
            result['gate_offsets'] = list(gate_offsets)

        return (score, result)
    else:
        return (1000.0, None)


# ============================================================================
# TWO-STAGE CMA-ES WITH EARLY STOPPING (shared utility functions)
# ============================================================================

def _run_cma_stage(
    individual, gate_config, initial_guess, bounds, initial_std,
    max_evaluations, num_workers, sim_time, dt, timeout_per_eval,
    gates_threshold, bspline_timing,
):
    """Run a single CMA-ES optimisation stage with early stopping.

    The wrapper auto-detects whether the solution vector contains timing
    params (len >= 7) so this works for both Stage 1 (4 params) and
    Stage 2 (7 params).

    Returns:
        Tuple of (best_result, total_evals, early_stopped, elapsed_seconds, iteration_fitnesses)
    """
    if not CMA_AVAILABLE:
        return None, 0, False, 0.0, []

    best_fitness = -float("inf")
    best_result = None
    total_evals = 0
    early_stopped = False
    iteration_fitnesses = []
    start_time = time.time()

    options = {
        "bounds": [list(b) for b in zip(*bounds)],
        "maxfevals": max_evaluations,
        "verb_disp": 0,
        "verb_log": 0,
        "tolx": 1e-11,
        "tolfun": 0,
        "tolfunhist": 0,
        "tolflatfitness": max_evaluations,
        "tolstagnation": max_evaluations,
    }

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            es = cma.CMAEvolutionStrategy(initial_guess, initial_std, options)

        executor = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
        try:
            while not es.stop():
                solutions = es.ask()

                if executor is not None:
                    eval_args = [
                        (sol, individual, gate_config, sim_time, dt, bspline_timing)
                        for sol in solutions
                    ]
                    future_to_sol = {
                        executor.submit(_evaluate_solution_wrapper, a): a[0]
                        for a in eval_args
                    }
                    results_dict = {}
                    for future in as_completed(future_to_sol):
                        sol = future_to_sol[future]
                        try:
                            score, result = future.result(timeout=timeout_per_eval)
                            results_dict[tuple(sol)] = score
                            if result is not None and result["fitness"] > best_fitness and not result["crashed"]:
                                best_fitness = result["fitness"]
                                best_result = result
                        except (TimeoutError, Exception):
                            results_dict[tuple(sol)] = 1000.0

                    fitness_values = [results_dict[tuple(sol)] for sol in solutions]
                else:
                    fitness_values = []
                    for sol in solutions:
                        score, result = _evaluate_solution_wrapper(
                            (sol, individual, gate_config, sim_time, dt, bspline_timing)
                        )
                        fitness_values.append(score)
                        if result is not None and result["fitness"] > best_fitness and not result["crashed"]:
                            best_fitness = result["fitness"]
                            best_result = result

                total_evals += len(solutions)
                es.tell(solutions, fitness_values)
                iteration_fitnesses.append(best_fitness if best_fitness > -float("inf") else 0.0)

                if best_result is not None and best_result["gates_passed"] >= gates_threshold:
                    early_stopped = True
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=False)

    except Exception as e:
        print(f"  CMA-ES error: {e}")

    elapsed = time.time() - start_time
    return best_result, total_evals, early_stopped, elapsed, iteration_fitnesses


def optimize_controller_with_early_stop(
    individual, gate_config, max_evaluations=200, num_workers=4,
    sim_time=20.0, dt=0.005, timeout_per_eval=30.0, gates_threshold=9,
    bspline_timing=None,
):
    """
    Two-stage CMA-ES optimisation with early stopping.

    Stage 1: Optimise gains + timing (7 params) with zero gate offsets.
    Stage 2: Optimise gains + timing + gate offsets (7 + n_gates*3 params),
             seeded from Stage 1 best.

    Returns dict with tuning results including eval count, timing, best gains,
    and best gate offsets.
    """
    _empty = {
        "gates_passed": 0, "n_evaluations": 0, "tuning_time_seconds": 0.0,
        "best_gains": None, "early_stopped": False, "distance_bonus": 0.0,
        "crashed": True, "best_bspline_timing": None, "best_gate_offsets": None,
    }
    if not CMA_AVAILABLE:
        return _empty

    if bspline_timing is None:
        bspline_timing = np.array([12.7, 4.6, 1.9])

    # Get gate offset bounds from BSplineGateTrajectory
    bspline_traj = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
    n_gates = bspline_traj.n_gates
    offset_bounds = bspline_traj.get_parameter_bounds_by_group()['gate_offsets']
    n_offset_params = n_gates * 3
    default_offsets = bspline_traj.get_default_parameters()[:n_offset_params]

    # Split budget: 40% Stage 1, 60% Stage 2
    stage1_evals = max(20, int(max_evaluations * 0.4))
    stage2_evals = max(20, max_evaluations - stage1_evals)

    # ------------------------------------------------------------------
    # Stage 1: Optimise gains + timing (7 params), zero gate offsets
    # ------------------------------------------------------------------
    stage1_guess = [14.3, 9.0, 2.9, -0.02, 12.7, 4.6, 1.9]
    stage1_bounds = [
        [10.0, 25.0],    # pos_P
        [0.1, 15.0],     # vel_P
        [0.1, 10.0],     # att_P
        [-1.0, -0.01],   # rate_P
        [5.0, 30.0],     # total_time
        [0.5, 10.0],     # velocity_scale
        [0.1, 5.0],      # startup_time
    ]

    best_result, evals1, early1, time1, _ = _run_cma_stage(
        individual, gate_config, stage1_guess, stage1_bounds, 1.5,
        stage1_evals, num_workers, sim_time, dt, timeout_per_eval,
        gates_threshold, bspline_timing,
    )

    if early1 and best_result is not None:
        # Already hit the gates threshold — skip Stage 2
        return {
            "gates_passed": best_result["gates_passed"],
            "n_evaluations": evals1,
            "tuning_time_seconds": round(time1, 2),
            "best_gains": best_result["gains"],
            "early_stopped": True,
            "distance_bonus": best_result.get("distance_bonus", 0.0),
            "crashed": best_result["crashed"],
            "best_bspline_timing": best_result.get("bspline_timing", list(bspline_timing)),
            "best_gate_offsets": list(default_offsets),
        }

    # ------------------------------------------------------------------
    # Stage 2: Optimise gains + timing + gate offsets (7 + n_gates*3 params)
    # ------------------------------------------------------------------
    if best_result is not None:
        g = best_result["gains"]
        stage2_gains_timing = [g["pos_P"], g["vel_P"], g["att_P"], g["rate_P"]]
        stage2_gains_timing += list(best_result.get("bspline_timing", bspline_timing))
    else:
        stage2_gains_timing = list(stage1_guess)

    stage2_guess = stage2_gains_timing + list(default_offsets)
    stage2_bounds = list(stage1_bounds) + [
        [float(default_offsets[i] + lo), float(default_offsets[i] + hi)]
        for i, (lo, hi) in enumerate(zip(offset_bounds[0], offset_bounds[1]))
    ]

    best2, evals2, early2, time2, _ = _run_cma_stage(
        individual, gate_config, stage2_guess, stage2_bounds, 0.3,
        stage2_evals, num_workers, sim_time, dt, timeout_per_eval,
        gates_threshold, bspline_timing,
    )

    # Pick best across both stages
    total_evals = evals1 + evals2
    total_time = time1 + time2

    if best2 is not None and (best_result is None or best2["fitness"] > best_result["fitness"]):
        best_result = best2

    if best_result is not None:
        return {
            "gates_passed": best_result["gates_passed"],
            "n_evaluations": total_evals,
            "tuning_time_seconds": round(total_time, 2),
            "best_gains": best_result["gains"],
            "early_stopped": early2,
            "distance_bonus": best_result.get("distance_bonus", 0.0),
            "crashed": best_result["crashed"],
            "best_bspline_timing": best_result.get("bspline_timing", list(bspline_timing)),
            "best_gate_offsets": best_result.get("gate_offsets", [0.0] * n_offset_params),
        }
    else:
        _empty["n_evaluations"] = total_evals
        _empty["tuning_time_seconds"] = round(total_time, 2)
        return _empty


# ============================================================================
# TWO-STAGE TUNER FOR SINGLE MORPHOLOGY (used by evolutionary loop)
# ============================================================================

def optimize_controller_for_morphology(individual, gate_config, max_evaluations=100,
                                      num_workers=None, sim_time=20.0, dt=0.005,
                                      timeout_per_eval=30.0, save_dir=None,
                                      bspline_timing=None):
    """
    Run 2-stage CMA-ES optimization to tune controller for a morphology.

    Stage 1: Optimise gains + timing (7 params) with zero gate offsets.
    Stage 2: Optimise gains + timing + gate offsets (7 + n_gates*3 params),
             seeded from Stage 1 best.

    Always runs both stages (no early stopping) for maximum quality.

    Args:
        individual: Evolved drone morphology (genome array)
        gate_config: Gate configuration class
        max_evaluations: Maximum CMA-ES evaluations (split across stages)
        num_workers: Number of parallel workers
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        timeout_per_eval: Timeout per evaluation in seconds
        save_dir: Directory to save results (optional)
        bspline_timing: Optional array of [total_time, velocity_scale, startup_time]

    Returns:
        dict with:
            - fitness: max gates passed (0 if failed)
            - best_gains: dict of best gains found
            - best_bspline_timing: best timing parameters
            - best_gate_offsets: best gate offset parameters
            - gates_passed: number of gates passed
            - success: True if optimization succeeded
    """
    if not CMA_AVAILABLE:
        print("ERROR: CMA-ES requires the 'cma' package")
        return {'fitness': 0, 'best_gains': None, 'gates_passed': 0, 'success': False}

    # Set up parallel workers
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)

    if bspline_timing is None:
        bspline_timing = np.array([12.7, 4.6, 1.9])

    # Get gate offset bounds from BSplineGateTrajectory
    bspline_traj = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
    n_gates = bspline_traj.n_gates
    offset_bounds = bspline_traj.get_parameter_bounds_by_group()['gate_offsets']
    n_offset_params = n_gates * 3
    default_offsets = bspline_traj.get_default_parameters()[:n_offset_params]

    # Split budget: 40% Stage 1, 60% Stage 2
    stage1_evals = max(20, int(max_evaluations * 0.4))
    stage2_evals = max(20, max_evaluations - stage1_evals)

    # No early stopping — always run both stages for max quality
    no_early_stop = 999999

    # ------------------------------------------------------------------
    # Stage 1: Optimise gains + timing (7 params), zero gate offsets
    # ------------------------------------------------------------------
    stage1_guess = [14.3, 9.0, 2.9, -0.02, 12.7, 4.6, 1.9]
    stage1_bounds = [
        [10.0, 25.0],    # pos_P
        [0.1, 15.0],     # vel_P
        [0.1, 10.0],     # att_P
        [-1.0, -0.01],   # rate_P
        [5.0, 30.0],     # total_time
        [0.5, 10.0],     # velocity_scale
        [0.1, 5.0],      # startup_time
    ]

    best_result, evals1, _, time1, stage1_fitnesses = _run_cma_stage(
        individual, gate_config, stage1_guess, stage1_bounds, 1.5,
        stage1_evals, num_workers, sim_time, dt, timeout_per_eval,
        no_early_stop, bspline_timing,
    )

    # ------------------------------------------------------------------
    # Stage 2: Optimise gains + timing + gate offsets (7 + n_gates*3 params)
    # ------------------------------------------------------------------
    if best_result is not None:
        g = best_result["gains"]
        stage2_gains_timing = [g["pos_P"], g["vel_P"], g["att_P"], g["rate_P"]]
        stage2_gains_timing += list(best_result.get("bspline_timing", bspline_timing))
    else:
        stage2_gains_timing = list(stage1_guess)

    stage2_guess = stage2_gains_timing + list(default_offsets)
    stage2_bounds = list(stage1_bounds) + [
        [float(default_offsets[i] + lo), float(default_offsets[i] + hi)]
        for i, (lo, hi) in enumerate(zip(offset_bounds[0], offset_bounds[1]))
    ]

    best2, evals2, _, time2, stage2_fitnesses = _run_cma_stage(
        individual, gate_config, stage2_guess, stage2_bounds, 0.3,
        stage2_evals, num_workers, sim_time, dt, timeout_per_eval,
        no_early_stop, bspline_timing,
    )

    # Pick best across both stages
    total_evals = evals1 + evals2
    total_time = time1 + time2
    best_score = -float('inf')

    if best_result is not None:
        best_score = best_result.get("fitness", 0)

    if best2 is not None and (best_result is None or best2["fitness"] > best_result["fitness"]):
        best_result = best2
        best_score = best2["fitness"]

    # Build combined learning curve (Stage 2 fitnesses carry forward Stage 1 best)
    s1_best = stage1_fitnesses[-1] if stage1_fitnesses else 0.0
    combined = list(stage1_fitnesses) + [max(s1_best, f) for f in stage2_fitnesses]

    # Save results if save_dir provided
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Always save genome
        np.save(os.path.join(save_dir, "genome.npy"), individual)

        # Save learning curve data with per-stage breakdown
        learning_curve_data = {
            "stage1": stage1_fitnesses,
            "stage2": stage2_fitnesses,
            "combined": combined,
        }
        with open(os.path.join(save_dir, "learning_curve.json"), 'w') as f:
            json.dump(learning_curve_data, f, indent=2)

        # Effective bspline_timing. Default gate_offsets come from the
        # B-spline's tension-based initialisation — this is what Stage 1
        # actually runs against (it passes gate_offsets=None to
        # simulate_with_gains). Saving zeros here would misrepresent the
        # trajectory Stage 1 winners were evaluated on.
        effective_timing = list(bspline_timing)
        effective_gate_offsets = list(default_offsets)

        # Enhanced tuning_results.json
        if best_result is not None:
            effective_timing = best_result.get('bspline_timing', effective_timing)
            if isinstance(effective_timing, np.ndarray):
                effective_timing = effective_timing.tolist()
            effective_gate_offsets = best_result.get('gate_offsets', effective_gate_offsets)
            if isinstance(effective_gate_offsets, np.ndarray):
                effective_gate_offsets = effective_gate_offsets.tolist()

            config = {
                'timestamp': datetime.now().isoformat(),
                'fitness': best_score,
                'gates_passed': best_result.get('gates_passed', 0),
                'distance_bonus': best_result.get('distance_bonus', 0.0),
                'gains': best_result['gains'],
                'bspline_timing': effective_timing,
                'gate_offsets': effective_gate_offsets,
                'brain': {
                    'gains': best_result['gains'],
                    'bspline_timing': effective_timing,
                    'gate_offsets': effective_gate_offsets,
                },
                'flight_time': best_result['flight_time'],
                'crashed': best_result['crashed'],
                'n_cma_iterations_stage1': len(stage1_fitnesses),
                'n_cma_iterations_stage2': len(stage2_fitnesses),
                'n_evaluations_stage1': evals1,
                'n_evaluations_stage2': evals2,
            }
        else:
            config = {
                'timestamp': datetime.now().isoformat(),
                'fitness': 0,
                'gates_passed': 0,
                'bspline_timing': list(bspline_timing),
                'gate_offsets': list(default_offsets),
                'crashed': True,
                'success': False,
                'n_cma_iterations_stage1': len(stage1_fitnesses),
                'n_cma_iterations_stage2': len(stage2_fitnesses),
                'n_evaluations_stage1': evals1,
                'n_evaluations_stage2': evals2,
            }

        with open(os.path.join(save_dir, "tuning_results.json"), 'w') as f:
            json.dump(config, f, indent=2)

        # Save plots (blueprint + learning curve)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer

            # Blueprint
            viz = DroneVisualizer()
            fig, _ = viz.plot_blueprint(individual, title="Morphology Blueprint")
            fig.savefig(os.path.join(save_dir, "blueprint.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Learning curve with stage boundary marker
            if len(combined) > 0:
                fig_lc, ax = plt.subplots(figsize=(8, 5))
                ax.plot(range(1, len(combined) + 1), combined, 'b-o', markersize=3)

                # Add vertical line at stage boundary
                if len(stage1_fitnesses) > 0 and len(stage2_fitnesses) > 0:
                    boundary = len(stage1_fitnesses) + 0.5
                    ax.axvline(x=boundary, color='red', linestyle='--', linewidth=1.5,
                               label='Stage 1 -> 2')
                    ax.legend()

                ax.set_xlabel('CMA-ES Iteration')
                ax.set_ylabel('Best Fitness (gates + distance bonus)')
                ax.set_title('Controller Tuning Learning Curve (2-Stage)')
                ax.grid(True, alpha=0.3)
                fig_lc.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_lc)
        except Exception as e:
            print(f"Warning: Could not save plots: {e}")

    # Return results
    if best_result is not None:
        return {
            'fitness': best_score,
            'best_gains': best_result['gains'],
            'best_bspline_timing': best_result.get('bspline_timing', list(bspline_timing)),
            'best_gate_offsets': best_result.get('gate_offsets', [0.0] * n_offset_params),
            'gates_passed': best_result['gates_passed'],
            'distance_bonus': best_result.get('distance_bonus', 0.0),
            'success': True
        }
    else:
        return {
            'fitness': 0,
            'best_gains': None,
            'best_bspline_timing': None,
            'best_gate_offsets': None,
            'gates_passed': 0,
            'distance_bonus': 0.0,
            'success': False
        }


# ============================================================================
# EVALUATOR FOR EVOLUTIONARY ALGORITHM
# ============================================================================

def evaluate_individual_with_tuning(individual, ind_save_dir, gate_cfg='circle',
                                    max_evals=100, num_workers=4, sim_time=20.0,
                                    dt=0.005, timeout=30.0, num=None):
    """
    Evaluate individual by tuning its controller via 2-stage CMA-ES.

    This is the main fitness function for evolution with Lee controller tuning.
    Stage 1 optimises gains + timing (7 params), Stage 2 adds gate offset
    control points (7 + n_gates*3 params) to refine the trajectory shape.

    Args:
        individual: Evolved drone morphology (genome array)
        ind_save_dir: Directory to save results for this individual
        gate_cfg: Gate configuration ('circle', 'figure8', 'slalom', 'backandforth')
        max_evals: Maximum CMA-ES evaluations (split across both stages)
        num_workers: Number of parallel workers for CMA-ES
        sim_time: Simulation time in seconds
        dt: Time step in seconds
        timeout: Timeout per evaluation in seconds
        num: Individual number (for logging)

    Returns:
        int: Number of gates passed (0 if tuning failed)
    """
    # Get gate configuration
    gate_config = GATE_CONFIGS[gate_cfg]

    # Run 2-stage CMA-ES optimization
    result = optimize_controller_for_morphology(
        individual=individual,
        gate_config=gate_config,
        max_evaluations=max_evals,
        num_workers=num_workers,
        sim_time=sim_time,
        dt=dt,
        timeout_per_eval=timeout,
        save_dir=ind_save_dir
    )

    # Return integer fitness (gates passed)
    return int(result['gates_passed'])
