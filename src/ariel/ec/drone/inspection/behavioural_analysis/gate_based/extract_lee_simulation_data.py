import json
import numpy as np

from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.ec.drone.evaluators.lee_tune_evaluator import simulate_with_gains


def extract_lee_simulation_data(genome, tuning_results_path, gate_cfg,
                                sim_time=20.0, dt=0.005):
    """
    Extract simulation data for a Lee-controller-tuned individual.

    Loads tuning results (gains, bspline_timing, gate_offsets), runs the
    simulation with trajectory recording, and returns data in the same
    format as extract_simulation_data (for RL individuals).

    Args:
        genome: Drone morphology array (N_arms, 6).
        tuning_results_path: Path to tuning_results.json.
        gate_cfg: Gate configuration name ('circle', 'figure8', 'slalom', 'backandforth').
        sim_time: Simulation time in seconds.
        dt: Simulation timestep in seconds.

    Returns:
        dict with keys:
            positions: (N, 3) array
            velocities: (N, 3) array
            angular_velocities: (N, 3) array
            gate_passes: (N,) boolean array
            actions: (N, num_motors) array normalised to [0, 1]
    """
    # Load tuning results
    with open(tuning_results_path, 'r') as f:
        tuning = json.load(f)

    gains = tuning['gains']
    bspline_timing = tuning.get('bspline_timing', None)
    gate_offsets = tuning.get('gate_offsets', None)

    # Legacy tuning_results.json files saved Stage-1 winners with
    # gate_offsets=[0.0]*N, but Stage 1 actually ran against the B-spline's
    # tension-based defaults (it passed gate_offsets=None). Remap all-zeros
    # to None so the trajectory reproduces what CMA-ES evaluated.
    if gate_offsets is not None and not np.any(np.asarray(gate_offsets)):
        gate_offsets = None

    gate_config = GATE_CONFIGS[gate_cfg]

    # Run simulation with trajectory recording
    result = simulate_with_gains(
        individual=genome,
        pos_gain=gains['pos_P'],
        vel_gain=gains['vel_P'],
        att_gain=gains['att_P'],
        rate_gain=gains['rate_P'],
        gate_config=gate_config,
        sim_time=sim_time,
        dt=dt,
        bspline_timing=bspline_timing,
        gate_offsets=gate_offsets,
        record_trajectory=True,
    )

    traj = result['trajectory']

    # Normalise motor commands to [0, 1] range
    # w_cmd values are angular velocities (rad/s); normalise per-motor by
    # mapping [0, max_observed] -> [0, 1] for visualisation consistency
    motor_cmds = traj['motor_commands']
    w_max = motor_cmds.max()
    if w_max > 0:
        actions = motor_cmds / w_max
    else:
        actions = np.zeros_like(motor_cmds)

    return {
        'positions': traj['positions'],
        'velocities': traj['velocities'],
        'angular_velocities': traj['angular_velocities'],
        'gate_passes': traj['gate_passes'],
        'actions': actions,
    }
