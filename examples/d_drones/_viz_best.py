"""Shared hover-simulation + video helper for evolution examples.

Imported by evolution scripts when ``--viz`` is passed. Not intended to be
run directly.

Public API:
    viz_best_from_db(db_path, video_path, duration, dt)   -- ARIEL SQLite EA runs
    viz_best_phenotype(phenotype, video_path, duration, dt) -- any (N,6) arm array
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _phenotype_to_propellers(phenotype: np.ndarray) -> list[dict]:
    """Convert (N, 6) arm array to propeller config list (NED frame).

    Column order: [magnitude, arm_yaw, arm_pitch, mot_pitch, mot_yaw, spin_dir]
    Matches the coordinate conventions in DroneGateEnv._convert_individual_to_propellers.
    """
    propellers = []
    for row in phenotype:
        magnitude, arm_yaw, arm_pitch, mot_pitch, mot_yaw, direction = row
        # Spherical → ENU cartesian
        enu_x = magnitude * np.cos(arm_pitch) * np.cos(arm_yaw)
        enu_y = magnitude * np.cos(arm_pitch) * np.sin(arm_yaw)
        enu_z = magnitude * np.sin(arm_pitch)
        # ENU → NED
        x, y, z = enu_y, enu_x, -enu_z
        # Thrust direction in NED
        sp, cp = np.sin(mot_pitch), np.cos(mot_pitch)
        sy, cy = np.sin(mot_yaw), np.cos(mot_yaw)
        propellers.append({
            "loc": [x, y, z],
            "dir": [-sy * sp, -cy * sp, cp, "cw" if direction > 0.5 else "ccw"],
            "propsize": 2,
        })
    return propellers


def _tune_hover_gains(
    propellers: list[dict],
    hover_pos: np.ndarray,
    dt: float = 0.005,
    n_eval_steps: int = 600,
) -> tuple[float, float]:
    """Find pos_P and vel_P gains that minimise z-axis hover error.

    Runs a short Nelder-Mead search (≈40 evaluations × n_eval_steps steps each).
    Starting point is taken from the Stage-1 gate-racing guess which has been
    empirically validated for small evolved morphologies.

    Returns (pos_P, vel_P) scalars to use for all three axes.
    """
    from scipy.optimize import minimize
    from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl
    from ariel.simulation.drone.controllers.utils.wind_model import Wind
    from ariel.simulation.drone.drone_interface import DroneInterface

    wind = Wind("None")
    sDes = np.zeros(19)
    sDes[:3] = hover_pos

    def _hover_cost(log_gains: np.ndarray) -> float:
        pos_g = float(np.exp(np.clip(log_gains[0], -2, 5)))
        vel_g = float(np.exp(np.clip(log_gains[1], -2, 5)))
        try:
            quad = DroneInterface(0, propellers=propellers)
            quad.drone_sim.set_state(
                position=hover_pos.copy(),
                velocity=np.zeros(3),
                attitude=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
            quad._update_state_variables()
            ctrl = LeeGeometricControl(
                quad, yawType=1, orient="NED", auto_scale_gains=True,
                pos_P_gain=np.array([pos_g] * 3),
                vel_P_gain=np.array([vel_g] * 3),
            )
            ctrl.controller(sDes, quad, "xyz_pos", dt)
            t = 0.0
            total_cost = 0.0
            for _ in range(n_eval_steps):
                t = quad.update(t, dt, ctrl.w_cmd, wind)
                ctrl.controller(sDes, quad, "xyz_pos", dt)
                err = quad.pos - hover_pos
                total_cost += float(np.dot(err, err))
            return total_cost / n_eval_steps
        except Exception:
            return 1e6

    # Stage-1 gate-racing best guess as warm start (empirically validated)
    x0 = np.array([np.log(14.3), np.log(9.0)])
    result = minimize(
        _hover_cost, x0, method="Nelder-Mead",
        options={"maxiter": 60, "xatol": 0.05, "fatol": 1e-3, "disp": False},
    )
    pos_P = float(np.exp(np.clip(result.x[0], -2, 5)))
    vel_P = float(np.exp(np.clip(result.x[1], -2, 5)))
    return pos_P, vel_P


def viz_best_phenotype(
    phenotype: np.ndarray,
    video_path: str | Path,
    duration: float = 10.0,
    dt: float = 0.005,
) -> None:
    """Simulate a hover for the given (N,6) phenotype and save to MP4."""
    import warnings
    warnings.filterwarnings("ignore", message=".*render_mode.*")

    import ariel.simulation.drone.controllers.utils as ctrl_utils
    from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl
    from ariel.simulation.drone.controllers.utils.wind_model import Wind
    from ariel.simulation.drone.drone_interface import DroneInterface

    propellers = _phenotype_to_propellers(phenotype)
    hover_pos = np.array([0.0, 0.0, -1.0])          # 1 m above origin in NED

    print("[viz] Tuning hover gains …", end=" ", flush=True)
    pos_P, vel_P = _tune_hover_gains(propellers, hover_pos, dt=dt)
    print(f"pos_P={pos_P:.2f}  vel_P={vel_P:.2f}")

    quad = DroneInterface(0, propellers=propellers)
    quad.drone_sim.set_state(
        position=hover_pos,
        velocity=np.zeros(3),
        attitude=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    quad._update_state_variables()

    ctrl = LeeGeometricControl(
        quad, yawType=1, orient="NED", auto_scale_gains=True,
        pos_P_gain=np.array([pos_P] * 3),
        vel_P_gain=np.array([vel_P] * 3),
    )
    wind = Wind("None")

    n_steps = int(duration / dt) + 1
    t_all    = np.zeros(n_steps)
    pos_all  = np.zeros((n_steps, 3))
    quat_all = np.zeros((n_steps, 4))
    sDes_all = np.zeros((n_steps, 19))
    sDes_all[:, :3] = hover_pos

    sDes = np.zeros(19)
    sDes[:3] = hover_pos
    ctrl.controller(sDes, quad, "xyz_pos", dt)

    t = 0.0
    for i in range(n_steps):
        t_all[i]    = t
        pos_all[i]  = quad.pos.copy()
        quat_all[i] = quad.quat.copy()
        t = quad.update(t, dt, ctrl.w_cmd, wind)
        ctrl.controller(sDes, quad, "xyz_pos", dt)

    arm_len = float(np.mean([np.linalg.norm(p["loc"]) for p in propellers]))
    params  = {"dxm": arm_len, "dym": arm_len, "dzm": 0.05}

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ctrl_utils.sameAxisAnimation(
        t_all, hover_pos.reshape(1, 3), pos_all, quat_all, sDes_all,
        dt, params, 0, 0, 1, "NED",
        save_path=str(video_path),
    )


def viz_best_from_db(
    db_path: str | Path,
    video_path: str | Path,
    duration: float = 10.0,
    dt: float = 0.005,
) -> None:
    """Load the highest-fitness individual from an ARIEL SQLite db and record a hover video."""
    from sqlalchemy import create_engine
    from sqlmodel import Session, select

    from ariel.body_phenotypes.drone.genome import deserialize_genome
    from ariel.ec.individual import Individual

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        stmt = (
            select(Individual)
            .where(Individual.requires_eval == False)  # noqa: E712
            .order_by(Individual.fitness_.desc())
            .limit(1)
        )
        best = session.exec(stmt).first()

    if best is None:
        print(f"[viz] No evaluated individuals found in {db_path}.")
        return

    genome = deserialize_genome(best.genotype)
    valid_mask = ~np.isnan(genome.arms[:, 0])
    phenotype  = genome.arms[valid_mask]

    print(
        f"[viz] Best: id={best.id}  fitness={best.fitness_:.4f}  "
        f"arms={int(valid_mask.sum())}  → {video_path}"
    )
    viz_best_phenotype(phenotype, video_path, duration=duration, dt=dt)
