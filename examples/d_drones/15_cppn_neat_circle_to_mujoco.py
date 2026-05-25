"""Evolve a drone morphology with CPPN-NEAT, then fly it through a circle in MuJoCo.

Pipeline:

    CPPNNeatDroneGenomeHandler population
        → evolve_neat (speciated NEAT with edit_distance fitness)
        → best CPPN.get_phenotype() → (max_narms, 6) spherical-angular array
        → spherical_angular_to_blueprint                  (decoder)
        → DroneBlueprint  (saved as JSON)
        → blueprint_to_propellers  + blueprint_to_mjspec  (two backends)
        → Python-sim Lee controller (NED) flies circle gates
        → kinematic playback in MuJoCo                    (MP4 / passive viewer)

The fitness for evolution is ``edit_distance`` — fast and self-contained
(``edit_distance`` would be a better proxy but pulls in the ``dronehover``
package which isn't on PyPI). It biases toward morphologies close to the
standard hexacopter shape; the "circle task" is then exercised in the
MuJoCo rollout stage on the evolved best individual.

Run:
    uv run examples/d_drones/15_cppn_neat_circle_to_mujoco.py
    uv run examples/d_drones/15_cppn_neat_circle_to_mujoco.py --pop 16 --gens 12 --view
    uv run examples/d_drones/15_cppn_neat_circle_to_mujoco.py --rollout-time 10 --altitude 1.5
"""
from __future__ import annotations

import argparse
import copy
import os
import time as _time
from pathlib import Path

import numpy as np
import mujoco
from rich.console import Console

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Evolution side
from ariel.ec.drone.genome_handlers.cppn_neat_genome_handler import (
    CPPNNeatDroneGenomeHandler,
)
from ariel.ec.drone.evaluators.unified_fitness import UnifiedFitness
from ariel.ec.drone.selectors.tournament import tournament_selection
from ariel.ec.drone.strategies import evolve_neat

# Blueprint side
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.backends import (
    blueprint_to_mjspec, blueprint_to_propellers,
)

# Simulation side
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.drone.drone_interface import DroneInterface
from ariel.simulation.drone.controllers.lee_control.lee_controller import (
    LeeGeometricControl,
)
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import (
    Trajectory,
)
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.drone.controllers.utils.wind_model import Wind
from ariel.utils.video_recorder import VideoRecorder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

console = Console()
parser = argparse.ArgumentParser(
    description="CPPN-NEAT → Blueprint → MuJoCo circle rollout"
)
# evolution params
parser.add_argument("--pop", type=int, default=12,
                    help="Population size (default 12, small for demo speed)")
parser.add_argument("--gens", type=int, default=8,
                    help="Number of NEAT generations (default 8)")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--min-arms", type=int, default=7)
parser.add_argument("--max-arms", type=int, default=8)
parser.add_argument("--num-segments", type=int, default=8,
                    help="CPPN sampling resolution (max_narms = arm slots)")
parser.add_argument("--seed", type=int, default=42)
# rollout params
parser.add_argument("--altitude", type=float, default=1.5,
                    help="Circle altitude in metres")
parser.add_argument("--rollout-time", type=float, default=12.0,
                    help="MuJoCo rollout duration in seconds")
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller inches used by Lee + MuJoCo")
# output
parser.add_argument("--view", action="store_true",
                    help="Launch passive viewer instead of writing MP4")
parser.add_argument("--out-dir", default="__data__/blueprint_demo/cppn_neat_circle")
args = parser.parse_args()

np.random.seed(args.seed)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Evolve a CPPN-NEAT population (fitness: edit_distance)
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],
    [-np.pi, np.pi],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [0, 1],
])

fitness_fn = UnifiedFitness(
    brain=None,
    fitness_mode="edit_distance",
    hover_gradient=False,
    per_individual_repair=False,
    is_indirect=True,                       # CPPN-NEAT is indirect
    handler_class=CPPNNeatDroneGenomeHandler,
    handler_kwargs={
        "num_segments": args.num_segments,
        "min_max_narms": (args.min_arms, args.max_arms),
        "parameter_limits": PARAMETER_LIMITS,
        "repair": True,
    },
    coordinate_system="spherical",
)

console.rule("[bold blue]CPPN-NEAT evolution")
console.log(f"pop={args.pop}  gens={args.gens}  fitness=edit_distance")

t_evo = _time.time()
all_individuals = evolve_neat(
    fitness_function=fitness_fn,
    population_size=args.pop,
    num_generations=args.gens,
    crossover_rate=0.75,
    parent_selection=tournament_selection,
    genome_handler=CPPNNeatDroneGenomeHandler,
    num_workers=args.workers,
    compatibility_threshold=3.0,
    target_species_count=3,
    log_dir=str(out_dir / "logs"),
    verbose=False,
)
console.log(f"Evolution done in {_time.time() - t_evo:.1f}s — "
            f"{len(all_individuals)} individuals evaluated")


# ---------------------------------------------------------------------------
# 2. Pick the best individual (last generation, highest fitness)
# ---------------------------------------------------------------------------

last_gen = int(all_individuals["generation"].max())
last_df = all_individuals[all_individuals["generation"] == last_gen]
best_row = last_df.sort_values("fitness", ascending=False).iloc[0]
best_cppn = best_row["genome"]

# Decode CPPN → spherical-angular phenotype array (max_narms, 6)
handler = CPPNNeatDroneGenomeHandler(
    genome=best_cppn,
    num_segments=args.num_segments,
    min_max_narms=(args.min_arms, args.max_arms),
    parameter_limits=PARAMETER_LIMITS,
)
phenotype = handler.get_phenotype()
n_active = int((~np.isnan(phenotype[:, 0])).sum())
console.log(f"Best (gen {last_gen}, id={best_row['id']}): "
            f"fitness={best_row['fitness']:.4f}  active_arms={n_active}")
np.save(out_dir / "best_phenotype.npy", phenotype)


# ---------------------------------------------------------------------------
# 3. Spherical-angular array → DroneBlueprint
# ---------------------------------------------------------------------------

bp = spherical_angular_to_blueprint(phenotype, propsize=args.prop_size)
bp_path = out_dir / "best_blueprint.json"
bp.save_json(bp_path)
console.log(f"Blueprint saved → {bp_path}")


# ---------------------------------------------------------------------------
# 4. Blueprint → DroneInterface (NED) + Lee + circle trajectory
#    (Lee's NED path is the validated one; we replay it in MuJoCo afterwards.)
# ---------------------------------------------------------------------------

propellers = blueprint_to_propellers(bp, convention="ned")
quad = DroneInterface(0, propellers=propellers)
wind = Wind("None")

ctrl = LeeGeometricControl(
    quad, yawType=1, orient="NED", auto_scale_gains=True,
    pos_P_gain=np.array([14.3, 14.3, 14.3]),
    vel_P_gain=np.array([9.0, 9.0, 9.0]),
)

gate_config = copy.deepcopy(GATE_CONFIGS["circle"])
gate_config.gate_pos = gate_config.gate_pos.copy()
gate_config.gate_pos[:, 2] = -float(args.altitude)   # NED: up = -z
gate_config.starting_pos = gate_config.starting_pos.copy()
gate_config.starting_pos[2] = -float(args.altitude)

traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                  gate_config=gate_config)

start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
_, vel_050, _ = traj.bspline_trajectory.evaluate(0.05)
initial_yaw = float(np.arctan2(vel_050[1], vel_050[0])) \
    if np.linalg.norm(vel_050[:2]) > 1e-3 else float(gate_config.gate_yaw[0])

quad.drone_sim.set_state(
    position=np.array(start_pos),
    velocity=np.zeros(3),
    attitude=np.array([0.0, 0.0, initial_yaw]),
    angular_velocity=np.zeros(3),
)
quad._update_state_variables()
sDes = traj.desiredState(0.0, args.dt, quad)
ctrl.controller(sDes, quad, traj.ctrlType, args.dt)


# ---------------------------------------------------------------------------
# 5. Roll out the Python NED sim, recording pose per step
# ---------------------------------------------------------------------------

Ts = float(args.dt)
Tf = float(args.rollout_time)
N = int(Tf / Ts) + 1
pos_log = np.zeros((N, 3))
quat_log = np.zeros((N, 4))

console.rule("[bold blue]Lee circle rollout (Python NED)")
i, t = 0, 0.0
t0 = _time.time()
while i < N and t < Tf:
    pos_log[i] = quad.pos
    quat_log[i] = quad.quat
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t = Ts * (i + 1)
    sDes = traj.desiredState(t, Ts, quad)
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)
    i += 1
pos_log = pos_log[:i]; quat_log = quat_log[:i]
console.log(f"  {i} steps in {_time.time() - t0:.2f}s  "
            f"alt range (NED z): [{pos_log[:,2].min():.2f}, {pos_log[:,2].max():.2f}]")

# NED→ENU
pos_enu = pos_log.copy(); pos_enu[:, 2] = -pos_enu[:, 2]
quat_enu = quat_log.copy(); quat_enu[:, 2] = -quat_enu[:, 2]


# ---------------------------------------------------------------------------
# 6. Build the MuJoCo body from the same blueprint, mass-matched to Lee
# ---------------------------------------------------------------------------

target_mass = float(quad.params["mB"])
n_arms_bp = sum(1 for _ in bp.children(bp.root_id))  # type: ignore[arg-type]
motor_mass_each = float(quad.drone_sim.config.propellers[0]["mass"])
# Average arm length (over the active arms) — used for the beam mass.
arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06
arm_mass_each = 0.034 * mean_arm_len
core_mass = max(1e-4, target_mass - n_arms_bp * (motor_mass_each + arm_mass_each))

drone_spec = blueprint_to_mjspec(
    bp,
    motor_mass=motor_mass_each,
    arm_mass=arm_mass_each,
    core_mass_override=core_mass,
    body_name="evolved_quad",
)
world = SimpleFlatWorld()
world.spawn(
    drone_spec,
    position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
    correct_collision_with_floor=False,
)
model = world.spec.compile()
data = mujoco.MjData(model)
model.opt.timestep = Ts


def set_pose(idx: int) -> None:
    p = pos_enu[idx]; q = quat_enu[idx]
    data.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
    data.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# 7. Render (passive viewer or MP4)
# ---------------------------------------------------------------------------

console.rule("[bold blue]MuJoCo playback")
if args.view:
    import mujoco.viewer
    console.log("Launching MuJoCo passive viewer …")
    set_pose(0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        idx = 0
        sim_start = _time.time()
        while viewer.is_running() and idx < len(pos_enu):
            step_start = _time.time()
            set_pose(idx)
            viewer.sync()
            slack = Ts - (_time.time() - step_start)
            if slack > 0:
                _time.sleep(slack)
            idx += 1
        console.log(f"Viewer done (wall {_time.time() - sim_start:.2f}s)")
else:
    mp4 = out_dir / "circle_flight.mp4"
    recorder = VideoRecorder(file_name=mp4.stem, output_folder=mp4.parent,
                             width=720, height=540, fps=30)
    steps_per_frame = max(1, int(round(1.0 / (recorder.fps * Ts))))
    t0 = _time.time()
    with mujoco.Renderer(model, width=recorder.width,
                         height=recorder.height) as renderer:
        for idx in range(0, len(pos_enu), steps_per_frame):
            set_pose(idx)
            renderer.update_scene(data)
            recorder.write(frame=renderer.render())
    recorder.release()
    console.log(f"Rendered in {_time.time() - t0:.2f}s")
    console.log(f"[bold green]Video: {mp4}")

console.log(f"Final pos (ENU): {pos_enu[-1].round(3).tolist()}")
