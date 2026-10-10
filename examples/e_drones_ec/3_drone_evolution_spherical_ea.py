"""Drone morphology evolution with ARIEL EA + spherical-angular direct encoding.

Mirrors the c_genotypes/3 pattern (ARIEL EA + direct genotype + MuJoCo
fitness) but for drones:

    SphericalAngularDroneGenomeHandler   (direct encoding)
        │
        ▼   ariel.ec EA pipeline (parent_tag → crossover → mutate → evaluate → truncate)
        │
        ▼   evaluate_drones_hover_mujoco
        │
        ▼   genotype → DroneBlueprint → MuJoCo (Lee→ctrl bridge) → hover fitness

Outputs: SQLite DB, best DroneBlueprint JSON, MP4 video, optional passive
viewer behind ``--visualize``. Single-process — for parallel evaluation
see ``6_drone_evolution_cppn_ea.py``.

Run::

    uv run examples/e_drones_ec/3_drone_evolution_spherical_ea.py
    uv run examples/e_drones_ec/3_drone_evolution_spherical_ea.py --pop 8 --budget 5
    uv run examples/e_drones_ec/3_drone_evolution_spherical_ea.py --no-visualize
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from rich.console import Console

from ariel.body_phenotypes.drone import (
    crossover_drones,
    deserialize_genome,
    evaluate_drones_hover_mujoco,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec import EA, Individual, Population
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

console = Console()
parser = argparse.ArgumentParser(
    description="Drone morphology evolution (spherical direct encoding, ARIEL EA, MuJoCo hover)",
)
parser.add_argument("--pop", type=int, default=8, help="Population size")
parser.add_argument("--budget", type=int, default=5, help="EA generations")
parser.add_argument("--min-arms", type=int, default=4, help="Minimum rotor arms")
parser.add_argument("--max-arms", type=int, default=6, help="Maximum rotor arms")
parser.add_argument("--dur", type=float, default=1.0, help="Hover-window seconds for fitness")
parser.add_argument("--warm-up", type=float, default=0.1, help="Warm-up seconds discarded from fitness log")
parser.add_argument("--target-alt", type=float, default=1.0, help="Hover target altitude (m, ENU)")
parser.add_argument(
    "--drift-weight", type=float, default=1.0,
    help="Weight for lateral drift in combined fitness",
)
parser.add_argument(
    "--tilt-weight", type=float, default=1.0,
    help="Weight for (1 - cos θ_tilt) in combined fitness",
)
parser.add_argument(
    "--ctrl-weight", type=float, default=0.05,
    help="Weight for mean(ctrl²) in combined fitness",
)
parser.add_argument("--prop-size", type=int, default=2, help="Propeller size (inches)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Render best individual: MP4 + passive viewer (--no-visualize to skip)",
)
parser.add_argument(
    "--viewer-duration", type=float, default=5.0,
    help="Seconds of hover to render in the post-evolution video / viewer",
)
args = parser.parse_args()


POP_SIZE = args.pop
BUDGET = args.budget
SEED = args.seed
np.random.seed(SEED)

DATA = Path.cwd() / "__data__" / Path(__file__).stem
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH = DATA / f"database_{RUN_ID}.db"

SPAWN_POSITION = (0.0, 0.0, float(args.target_alt))


# ---------------------------------------------------------------------------
# Genome handler (direct, fixed-size spherical-angular)
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],            # arm magnitude (m)
    [-np.pi, np.pi],          # arm azimuth
    [-np.pi / 2, np.pi / 2],  # arm elevation
    [-np.pi, np.pi],          # motor disc azimuth
    [-np.pi, np.pi],          # motor disc pitch
    [0, 1],                   # propeller spin direction
])

template_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(args.min_arms, args.max_arms),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=0.1,
    bilateral_plane_for_symmetry=None,
    repair=False,
    rnd=np.random.default_rng(SEED),
)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

console.rule("[bold blue]Drone Spherical-Angular Evolution (ARIEL EA + MuJoCo hover)")
console.log(
    f"pop={POP_SIZE}  budget={BUDGET}  arms=[{args.min_arms}, {args.max_arms}]  "
    f"dur={args.dur}s  target_alt={args.target_alt}m  prop_size={args.prop_size}",
)
console.log(f"DB → {DB_PATH}")


# ---------------------------------------------------------------------------
# Initial population (init + evaluate pre-loop)
# ---------------------------------------------------------------------------

initial_pop = Population([Individual() for _ in range(POP_SIZE)])

init_op = initialize_drones(template_handler=template_handler)
eval_op = evaluate_drones_hover_mujoco(
    decoder="spherical",
    decoder_kwargs={"propsize": args.prop_size},
    duration=args.dur,
    warm_up=args.warm_up,
    target_position=SPAWN_POSITION,
    n_workers=1,
    drift_weight=args.drift_weight,
    tilt_weight=args.tilt_weight,
    ctrl_weight=args.ctrl_weight,
)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop if np.isfinite(ind.fitness_ or np.inf)]
if finite0:
    console.log(f"Initial fitness — min={min(finite0):.4f}  mean={np.mean(finite0):.4f}")


# ---------------------------------------------------------------------------
# Generation pipeline (lower-is-better hover fitness; survivors are top-n minimum)
# ---------------------------------------------------------------------------

generation_ops = [
    parent_tag(n=POP_SIZE),
    crossover_drones(template_handler=template_handler),
    mutate_drones(template_handler=template_handler),
    eval_op,
    truncation_select(n=POP_SIZE),
]

ea = EA(
    population=initial_pop,
    operations=generation_ops,
    num_steps=BUDGET,
    db_file_path=DB_PATH,
    db_handling="delete",
)

console.rule("[bold green]Evolving")
ea.run()


# ---------------------------------------------------------------------------
# Pick best (minimisation → smallest fitness is best)
# ---------------------------------------------------------------------------

ea.fetch_population(only_alive=False, requires_eval=False)
all_evaluated = ea.population
finite = [ind for ind in all_evaluated if np.isfinite(ind.fitness_ or np.inf)]
if not finite:
    console.log("[red]No evaluated individuals with finite fitness — aborting visualization.[/red]")
    sys.exit(0)
best = sorted(finite, key=lambda i: i.fitness_)[0]

console.rule("[bold cyan]Best individual")
console.log(f"  id={best.id}  fitness={best.fitness_:.4f}  born={best.time_of_birth}")

best_genome = deserialize_genome(best.genotype)
valid_mask = ~np.isnan(best_genome.arms[:, 0])
console.log(f"  active arms: {int(valid_mask.sum())} / {best_genome.arms.shape[0]}")

best_bp = spherical_angular_to_blueprint(best_genome.arms, propsize=args.prop_size)
bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
best_bp.save_json(bp_path)
console.log(f"  blueprint → {bp_path}")


# ---------------------------------------------------------------------------
# Replay (MP4 + optional passive viewer)
# ---------------------------------------------------------------------------

if args.visualize:
    import mujoco

    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
    from ariel.simulation.drone.controllers.lee_control.lee_controller import (
        LeeGeometricControl,
    )
    from ariel.simulation.drone.controllers.lee_control.mujoco_bridge import (
        LeeMujocoHoverBridge,
        spawn_blueprint_in_world,
    )
    from ariel.simulation.drone.drone_interface import DroneInterface
    from ariel.utils.video_recorder import VideoRecorder

    propellers = blueprint_to_propellers(best_bp, convention="ned")
    quad = DroneInterface(0, propellers=propellers)
    lee_ctrl = LeeGeometricControl(
        quad, yawType=1, orient="NED", auto_scale_gains=True,
        pos_P_gain=np.array([14.3, 14.3, 14.3]),
        vel_P_gain=np.array([9.0, 9.0, 9.0]),
    )
    spawned = spawn_blueprint_in_world(
        best_bp, propellers=propellers,
        target_mass=float(quad.params["mB"]),
        spawn_position=SPAWN_POSITION,
    )
    bridge = LeeMujocoHoverBridge(
        quad=quad, lee_ctrl=lee_ctrl,
        model=spawned.model, data=spawned.data,
        max_thrust_per_motor=spawned.max_thrust_per_motor,
        target_position_enu=SPAWN_POSITION,
    )

    # MP4
    videos_dir = DATA / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder(
        file_name=f"best_hover_{RUN_ID}",
        output_folder=videos_dir,
    )
    bridge.reset_pose()
    steps_per_frame = max(1, int(round(1.0 / (spawned.model.opt.timestep * video.fps))))
    total_steps = int(args.viewer_duration / spawned.model.opt.timestep)
    console.log(f"Recording MP4 ({args.viewer_duration}s) …")
    with mujoco.Renderer(spawned.model, width=video.width, height=video.height) as renderer:
        for step_i in range(total_steps):
            bridge.step()
            mujoco.mj_step(spawned.model, spawned.data)
            if step_i % steps_per_frame == 0:
                renderer.update_scene(spawned.data)
                video.write(renderer.render())
    video.release()
    console.log(f"  MP4 → {videos_dir}")

    # Passive viewer (skip on macOS / no passive support)
    if sys.platform != "darwin" and hasattr(mujoco.viewer, "launch_passive"):
        bridge.reset_pose()
        console.log("Launching passive viewer (close window to exit) …")
        with mujoco.viewer.launch_passive(spawned.model, spawned.data) as v:
            t_start = time.time()
            while v.is_running() and (time.time() - t_start) < args.viewer_duration:
                step_start = time.time()
                bridge.step()
                mujoco.mj_step(spawned.model, spawned.data)
                v.sync()
                slack = spawned.model.opt.timestep - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)


console.rule("[bold]Done")
