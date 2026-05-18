"""Curriculum-based CMA-ES tuning of the Lee geometric controller for gate racing.

Optimises controller gains and B-spline trajectory parameters for a 2-inch
quad in three progressive stages:

    Stage 1 (gains only)  →  Stage 2 (gains + timing)  →  Stage 3 (full)

Requires the ``cma`` package (``uv pip install cma``).

Run:
    # All three stages automatically:
    python examples/d_drones/4_tune_lee_controller.py --stage all --gates figure8

    # Individual stages:
    python examples/d_drones/4_tune_lee_controller.py --stage 1 --gates figure8 --max-evals 200
    python examples/d_drones/4_tune_lee_controller.py --stage 2 --gates figure8 \\
        --load-prev __data__/lee_tuning/stage1_best.json --max-evals 300
    python examples/d_drones/4_tune_lee_controller.py --stage 3 --gates figure8 \\
        --load-prev __data__/lee_tuning/stage2_best.json --max-evals 500

After tuning, visualise with:
    python examples/d_drones/3_simulate_lee.py \\
        --gates figure8 --bspline-config __data__/lee_tuning/stage3_best.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

# Add the d_drones directory to sys.path so _ctrl_helpers is importable
sys.path.insert(0, str(Path(__file__).parent))
from _ctrl_helpers import ARM_LENGTH, PROP_SIZE, GateChecker, create_2inch_quad

from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl
from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import (
    BSplineGateTrajectory,
)
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import Trajectory
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.drone.controllers.utils.wind_model import Wind

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    cma = None
    CMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Curriculum CMA-ES tuning for Lee controller + B-spline trajectory",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--stage", required=True,
                    help="Curriculum stage: 1, 2, 3, or all")
parser.add_argument("--gates", required=True,
                    choices=["figure8", "circle", "slalom", "backandforth"],
                    help="Gate configuration")
parser.add_argument("--load-prev",
                    help="JSON from a previous stage to seed this stage")
parser.add_argument("--max-evals", type=int, default=200,
                    help="Max CMA-ES evaluations (default 200)")
parser.add_argument("--max-evals-1", type=int, default=None,
                    help="Override --max-evals for Stage 1")
parser.add_argument("--max-evals-2", type=int, default=None,
                    help="Override --max-evals for Stage 2")
parser.add_argument("--max-evals-3", type=int, default=None,
                    help="Override --max-evals for Stage 3")
parser.add_argument("--workers", type=int, default=None,
                    help="Parallel workers (default: cpu_count // 2)")
parser.add_argument("--time", type=float, default=20.0,
                    help="Simulation time per evaluation in seconds (default 20)")
parser.add_argument("--dt", type=float, default=0.005,
                    help="Simulation timestep in seconds (default 0.005)")
parser.add_argument("--output", default="__data__/lee_tuning",
                    help="Output directory (default: __data__/lee_tuning)")
parser.add_argument("--timeout", type=float, default=30.0,
                    help="Timeout per evaluation in seconds (default 30)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def simulate_bspline(
    pos_gain: float, vel_gain: float, att_gain: float, rate_gain: float,
    bspline_params: np.ndarray,
    gate_config,
    sim_time: float = 20.0,
    dt: float = 0.005,
    verbose: bool = False,
) -> dict:
    """Run one Lee + B-spline evaluation; return result dict."""
    try:
        quad = create_2inch_quad()
        ctrl = LeeGeometricControl(
            quad, yawType=1, orient="NED",
            auto_scale_gains=True,           # att/rate auto-derived from inertia
            pos_P_gain=np.array([pos_gain] * 3),
            vel_P_gain=np.array([vel_gain] * 3),
            # att_P_gain and rate_P_gain intentionally omitted → auto-scaled
        )

        bspline_traj = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
        bspline_traj.set_parameters(bspline_params)

        start_pos, _, _ = bspline_traj.evaluate(0.0)
        _, vel_050, _ = bspline_traj.evaluate(0.05)
        if np.linalg.norm(vel_050[:2]) > 0.001:
            initial_yaw = np.arctan2(vel_050[1], vel_050[0])
        else:
            initial_yaw = gate_config.gate_yaw[0]
        initial_euler = np.array([0.0, 0.0, initial_yaw])
        quad.drone_sim.set_state(
            position=start_pos, velocity=np.zeros(3),
            attitude=initial_euler, angular_velocity=np.zeros(3),
        )

        from ariel.simulation.drone.controllers.trajectory_generation.trajectory import Trajectory
        traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                          gate_config=gate_config)
        traj.bspline_trajectory = bspline_traj

        wind = Wind("None")
        gate_checker = GateChecker(gate_config.gate_pos, gate_config.gate_yaw,
                                   gate_config.gate_size)

        sDes = traj.desiredState(0, dt, quad)
        ctrl.controller(sDes, quad, "xyz_pos", dt)

        t = 0.0
        crashed = False
        num_steps = int(sim_time / dt)
        for step in range(num_steps):
            quad.update(t, dt, ctrl.w_cmd, wind)
            t = dt * (step + 1)
            sDes = traj.desiredState(t, dt, quad)
            ctrl.controller(sDes, quad, "xyz_pos", dt)
            gate_checker.check_gate_passing(quad.pos)
            if np.linalg.norm(quad.pos - sDes[:3]) > 10.0:
                crashed = True
                break

        distance_bonus = gate_checker.get_normalized_distance_to_next_gate(quad.pos)
        return {
            "gates_passed": gate_checker.gates_passed,
            "distance_bonus": distance_bonus,
            "crashed": crashed,
            "flight_time": t,
        }
    except Exception as exc:
        if verbose:
            print(f"Simulation error: {exc}")
        return {"gates_passed": 0, "distance_bonus": 0.0, "crashed": True, "flight_time": 0.0}


def _eval_worker(args_tuple) -> dict:
    (params, stage, gate_config, sim_time, dt,
     fixed_bspline, fixed_gains, fixed_timing, fixed_offsets,
     n_offset_params) = args_tuple

    # att/rate are auto-scaled — only pos and vel come from the optimizer
    if stage == 1:
        pos_g, vel_g = params[:2]
        bspline_params = fixed_bspline
    elif stage == 2:
        pos_g, vel_g = params[:2]
        timing = params[2:5]
        bspline_params = np.concatenate([fixed_offsets, timing])
    else:  # stage 3
        pos_g, vel_g = params[:2]
        timing = params[2:5]
        offsets = params[5:5 + n_offset_params]
        bspline_params = np.concatenate([offsets, timing])

    # att_gain and rate_gain are ignored (auto-scaled in simulate_bspline)
    result = simulate_bspline(pos_g, vel_g, att_gain=0.0, rate_gain=0.0,
                              bspline_params=bspline_params, gate_config=gate_config,
                              sim_time=sim_time, dt=dt)
    gates = result["gates_passed"]
    bonus = result["distance_bonus"]
    return {"score": gates + bonus, **result, "params": params.tolist()}


# ---------------------------------------------------------------------------
# CurriculumTuner
# ---------------------------------------------------------------------------

class CurriculumTuner:
    def __init__(self, gate_config, stage: int, sim_time: float = 20.0,
                 dt: float = 0.005, output_dir: str = "__data__/lee_tuning") -> None:
        self.gate_config = gate_config
        self.stage = stage
        self.sim_time = sim_time
        self.dt = dt
        self.output_dir = output_dir
        self.best_score = -float("inf")
        self.best_params: dict | None = None
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        tmpl = BSplineGateTrajectory(gate_config, gate_offset_scale=0.5)
        self.fixed_bspline = tmpl.get_default_parameters()
        self.fixed_offsets = tmpl.get_gate_offset_parameters()
        self.n_offset_params = tmpl.get_gate_offset_count()
        self.fixed_timing = np.array([12.7, 4.6, 1.9])
        self.fixed_bspline[-3:] = self.fixed_timing
        # Only pos/vel are optimised; att/rate are auto-scaled from drone inertia
        self.fixed_gains = [14.3, 9.0]   # [pos_P, vel_P]

    def load_previous_stage(self, config_path: str) -> None:
        print(f"Loading previous stage from: {config_path}")
        with open(config_path) as f:
            cfg = json.load(f)
        if "bspline_params" in cfg:
            bp = np.array(cfg["bspline_params"])
            self.fixed_bspline = bp
            self.fixed_offsets = bp[: self.n_offset_params]
            self.fixed_timing = bp[-3:]
        if "gains" in cfg:
            g = cfg["gains"]
            self.fixed_gains = [g["pos_P"], g["vel_P"]]

    def _build_config(self) -> dict:
        tmpl = BSplineGateTrajectory(self.gate_config, gate_offset_scale=0.5)
        bounds_by_group = tmpl.get_parameter_bounds_by_group()
        gate_lower, gate_upper = bounds_by_group["gate_offsets"]

        if self.stage == 1:
            ig = list(self.fixed_gains)
            bounds = [[1.0, 20.0], [1.0, 20.0]]  # pos_P, vel_P only
            return {"initial_guess": ig, "bounds": bounds, "initial_std": 1.5,
                    "param_count": 2, "description": "pos/vel gains (2 params; att/rate auto-scaled)"}
        elif self.stage == 2:
            ig = list(self.fixed_gains) + list(self.fixed_timing)
            bounds = [
                [1.0, 20.0], [1.0, 20.0],   # pos_P, vel_P
                [8.0, 18.0], [1.0, 10.0], [0.1, 5.0],  # timing
            ]
            return {"initial_guess": ig, "bounds": bounds, "initial_std": 1.5,
                    "param_count": 5, "description": "pos/vel gains (2) + Timing (3)"}
        else:  # stage 3
            ig = list(self.fixed_gains) + list(self.fixed_timing) + list(self.fixed_offsets)
            bounds = [
                [1.0, 20.0], [1.0, 20.0],   # pos_P, vel_P
                [8.0, 18.0], [1.0, 10.0], [0.1, 5.0],  # timing
            ]
            for lo, hi in zip(gate_lower, gate_upper):
                bounds.append([lo, hi])
            n = 5 + self.n_offset_params
            return {"initial_guess": ig, "bounds": bounds, "initial_std": 0.1,
                    "param_count": n,
                    "description": f"pos/vel (2) + Timing (3) + Gate offsets ({self.n_offset_params})"}

    def _score(self, params: np.ndarray) -> float:
        r = _eval_worker((
            params, self.stage, self.gate_config, self.sim_time, self.dt,
            self.fixed_bspline, self.fixed_gains, self.fixed_timing,
            self.fixed_offsets, self.n_offset_params,
        ))
        score = r["score"]
        if score > self.best_score:
            self.best_score = score
            pos_g, vel_g = params[:2]
            self.best_params = {
                **r,
                "gains": {"pos_P": pos_g, "vel_P": vel_g},
                "bspline_params": (
                    self.fixed_bspline.tolist() if self.stage == 1 else
                    np.concatenate([self.fixed_offsets, params[2:5]]).tolist() if self.stage == 2 else
                    np.concatenate([params[5:5 + self.n_offset_params], params[2:5]]).tolist()
                ),
            }
        return -score  # CMA minimises

    def run_optimization(self, max_evaluations: int = 200,
                         num_workers: int | None = None,
                         timeout_per_eval: float = 30.0) -> None:
        if not CMA_AVAILABLE:
            print("ERROR: CMA-ES requires the 'cma' package (uv pip install cma)")
            return

        cfg = self._build_config()
        num_workers = num_workers or max(1, multiprocessing.cpu_count() // 2)

        print(f"\n{'=' * 70}")
        print(f"CURRICULUM STAGE {self.stage} — CMA-ES OPTIMISATION")
        print(f"{'=' * 70}")
        print(f"Gate config     : {self.gate_config.__name__}")
        print(f"Optimising      : {cfg['description']}  ({cfg['param_count']} params)")
        print(f"Max evaluations : {max_evaluations}  workers={num_workers}")
        print(f"{'=' * 70}\n")

        ig = np.array(cfg["initial_guess"], dtype=float)
        bounds_lo = [b[0] for b in cfg["bounds"]]
        bounds_hi = [b[1] for b in cfg["bounds"]]

        options = {
            "maxfevals": max_evaluations,
            "bounds": [bounds_lo, bounds_hi],
            "verbose": -9,
            "tolx": 1e-4,
            "tolfun": 1e-4,
        }

        t0 = time.time()
        try:
            es = cma.CMAEvolutionStrategy(ig, cfg["initial_std"], options)
            iteration = 0
            while not es.stop():
                candidates = es.ask()
                fitnesses = [self._score(np.array(c)) for c in candidates]
                es.tell(candidates, fitnesses)
                iteration += 1
                if iteration % 10 == 0:
                    print(f"  iter {iteration:4d}  evals {es.result.evaluations:5d}  "
                          f"best={self.best_score:.3f}")
        except KeyboardInterrupt:
            print("\nInterrupted — saving current best …")
        finally:
            elapsed = time.time() - t0
            print(f"\nStage {self.stage} done in {elapsed / 60:.1f} min  "
                  f"({es.result.evaluations} evals)  best={self.best_score:.3f}")
            self._save_results()

    def _save_results(self) -> None:
        if self.best_params is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "stage": self.stage,
            "timestamp": datetime.now().isoformat(),
            "fitness": self.best_score,
            "gates_passed": self.best_params.get("gates_passed", 0),
            "distance_bonus": self.best_params.get("distance_bonus", 0.0),
            "gains": self.best_params["gains"],
            "bspline_params": self.best_params["bspline_params"],
            "flight_time": self.best_params.get("flight_time", 0.0),
            "crashed": self.best_params.get("crashed", True),
        }
        best_path = os.path.join(self.output_dir, f"stage{self.stage}_best.json")
        with open(best_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nStage {self.stage} best config saved: {best_path}")
        if self.stage < 3:
            print(f"  To continue: --stage {self.stage + 1} --load-prev {best_path}")
        else:
            print(f"  To visualise: python examples/d_drones/3_simulate_lee.py "
                  f"--gates {args.gates} --bspline-config {best_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

gate_config = GATE_CONFIGS[args.gates]
output_dir = args.output

if args.stage.lower() == "all":
    evals = [
        args.max_evals_1 or args.max_evals,
        args.max_evals_2 or args.max_evals,
        args.max_evals_3 or args.max_evals,
    ]
    prev_path = args.load_prev
    tuners = []
    for stage_num, max_ev in zip([1, 2, 3], evals):
        print(f"\n{'#' * 70}\n# STAGE {stage_num}\n{'#' * 70}")
        tuner = CurriculumTuner(gate_config, stage=stage_num,
                                sim_time=args.time, dt=args.dt,
                                output_dir=output_dir)
        if prev_path:
            tuner.load_previous_stage(prev_path)
        tuner.run_optimization(max_evaluations=max_ev, num_workers=args.workers,
                               timeout_per_eval=args.timeout)
        tuners.append(tuner)
        prev_path = os.path.join(output_dir, f"stage{stage_num}_best.json")
        if not os.path.exists(prev_path):
            print(f"Stage {stage_num} did not produce results — stopping.")
            break

    print("\n" + "=" * 70)
    print("ALL STAGES COMPLETE")
    print("=" * 70)
    for i, t in enumerate(tuners, 1):
        print(f"  Stage {i}: {t.best_score:.3f}")

else:
    stage_num = int(args.stage)
    tuner = CurriculumTuner(gate_config, stage=stage_num,
                            sim_time=args.time, dt=args.dt,
                            output_dir=output_dir)
    if args.load_prev:
        tuner.load_previous_stage(args.load_prev)
    tuner.run_optimization(max_evaluations=args.max_evals,
                           num_workers=args.workers,
                           timeout_per_eval=args.timeout)
