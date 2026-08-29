"""Drone morphology evolution with ARIEL EA + CPPN-NEAT indirect encoding.

Mirrors the c_genotypes/6 pattern (ARIEL EA + indirect genotype +
per-individual MuJoCo fitness + multiprocessing) but for drones:

    CPPNNeatDroneGenomeHandler   (CPPN-NEAT indirect encoding)
        │
        ▼   ariel.ec EA pipeline (parent_tag → crossover → mutate → evaluate → truncate)
        │
        ▼   evaluate_drones_hover_mujoco  (multiprocessing.Pool, 'spawn' ctx)
        │
        ▼   CPPN → phenotype → DroneBlueprint → MuJoCo (Lee→ctrl bridge) → hover fitness

Outputs: SQLite DB, best CPPN JSON, best DroneBlueprint JSON, MP4 video,
optional passive viewer behind ``--visualize``.

Run::

    uv run examples/e_drones_ec/6_drone_evolution_cppn_ea.py
    uv run examples/e_drones_ec/6_drone_evolution_cppn_ea.py --pop 8 --budget 5 --workers 4
    uv run examples/e_drones_ec/6_drone_evolution_cppn_ea.py --no-visualize
"""
from __future__ import annotations

import argparse
import json
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
    crossover_cppn_drones,
    deserialize_cppn_genome,
    evaluate_drones_hover_mujoco,
    initialize_cppn_drones,
    mutate_cppn_drones,
    parent_tag,
    truncation_select,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec import EA, Individual, Population
from ariel.ec.drone.genome_handlers.cppn_neat_genome_handler import (
    CPPNNeatDroneGenomeHandler,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone morphology evolution (CPPN-NEAT indirect encoding, ARIEL EA, MuJoCo hover)",
    )
    parser.add_argument("--pop", type=int, default=8, help="Population size")
    parser.add_argument("--budget", type=int, default=5, help="EA generations")
    parser.add_argument("--min-arms", type=int, default=4, help="Minimum rotor arms")
    parser.add_argument("--max-arms", type=int, default=6, help="Maximum rotor arms")
    parser.add_argument(
        "--num-segments", type=int, default=8,
        help="CPPN sampling resolution per arm slot",
    )
    parser.add_argument("--dur", type=float, default=1.0, help="Hover-window seconds for fitness")
    parser.add_argument("--warm-up", type=float, default=0.1, help="Warm-up seconds discarded from fitness log")
    parser.add_argument("--target-alt", type=float, default=1.0, help="Hover target altitude (m, ENU)")
    parser.add_argument("--drift-weight", type=float, default=1.0)
    parser.add_argument("--tilt-weight", type=float, default=1.0)
    parser.add_argument("--ctrl-weight", type=float, default=0.05)
    parser.add_argument("--prop-size", type=int, default=2, help="Propeller size (inches)")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2),
        help="Parallel evaluation worker processes (1 = single-process)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render best individual: MP4 + passive viewer (--no-visualize to skip)",
    )
    parser.add_argument("--viewer-duration", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    POP_SIZE = args.pop
    BUDGET = args.budget
    SEED = args.seed
    np.random.seed(SEED)

    DATA = Path.cwd() / "__data__" / Path(__file__).stem
    DATA.mkdir(parents=True, exist_ok=True)
    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    DB_PATH = DATA / f"database_{RUN_ID}.db"

    SPAWN_POSITION = (0.0, 0.0, float(args.target_alt))

    PARAMETER_LIMITS = np.array([
        [0.055, 0.17],
        [-np.pi, np.pi],
        [-np.pi / 2, np.pi / 2],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [0, 1],
    ])

    HANDLER_KWARGS = {
        "num_segments": args.num_segments,
        "min_max_narms": (args.min_arms, args.max_arms),
        "parameter_limits": PARAMETER_LIMITS,
        "repair": True,
    }

    template_handler = CPPNNeatDroneGenomeHandler(
        rng=np.random.default_rng(SEED),
        **HANDLER_KWARGS,
    )

    console.rule("[bold blue]Drone CPPN-NEAT Evolution (ARIEL EA + MuJoCo hover)")
    console.log(
        f"pop={POP_SIZE}  budget={BUDGET}  arms=[{args.min_arms}, {args.max_arms}]  "
        f"segs={args.num_segments}  workers={args.workers}  dur={args.dur}s",
    )
    console.log(f"DB → {DB_PATH}")

    initial_pop = Population([Individual() for _ in range(POP_SIZE)])

    init_op = initialize_cppn_drones(template_handler=template_handler)
    eval_op = evaluate_drones_hover_mujoco(
        decoder="cppn",
        decoder_kwargs={
            "propsize": args.prop_size,
            "handler_kwargs": HANDLER_KWARGS,
        },
        duration=args.dur,
        warm_up=args.warm_up,
        target_position=SPAWN_POSITION,
        n_workers=args.workers,
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

    generation_ops = [
        parent_tag(n=POP_SIZE),
        crossover_cppn_drones(template_handler=template_handler),
        mutate_cppn_drones(template_handler=template_handler),
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

    ea.fetch_population(only_alive=False, requires_eval=False)
    all_evaluated = ea.population
    finite = [ind for ind in all_evaluated if np.isfinite(ind.fitness_ or np.inf)]
    if not finite:
        console.log("[red]No evaluated individuals with finite fitness — aborting visualization.[/red]")
        return
    best = sorted(finite, key=lambda i: i.fitness_)[0]

    console.rule("[bold cyan]Best individual")
    console.log(f"  id={best.id}  fitness={best.fitness_:.4f}  born={best.time_of_birth}")

    best_net = deserialize_cppn_genome(best.genotype)
    best_handler = CPPNNeatDroneGenomeHandler(genome=best_net, **HANDLER_KWARGS)
    best_phenotype = best_handler.get_phenotype()
    n_active = int((~np.isnan(best_phenotype[:, 0])).sum())
    console.log(f"  decoded active arms: {n_active} / {best_phenotype.shape[0]}")

    best_bp = spherical_angular_to_blueprint(best_phenotype, propsize=args.prop_size)
    bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
    cppn_path = DATA / f"best_cppn_{RUN_ID}.json"
    best_bp.save_json(bp_path)
    cppn_path.write_text(json.dumps(best.genotype, indent=2), encoding="utf-8")
    console.log(f"  CPPN genome → {cppn_path}")
    console.log(f"  blueprint   → {bp_path}")

    if args.visualize and n_active > 0:
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
        if not propellers:
            console.log("[yellow]Best individual has no decoded propellers — skipping visualisation.[/yellow]")
            return
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

        videos_dir = DATA / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        video = VideoRecorder(file_name=f"best_hover_{RUN_ID}", output_folder=videos_dir)
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


if __name__ == "__main__":
    main()
