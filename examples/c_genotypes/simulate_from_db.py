"""Load best individual from an EA database and simulate it.

Example
-------
uv run python examples/c_genotypes/simulate_from_db.py \
  --db-path __data__/2_body_brain_evolution_tree_fast/database.db \
  --viewer launcher --dur 20
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal, cast

import mujoco
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from sqlalchemy import create_engine
from sqlmodel import Session, col, select

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.a001 import Individual
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

console = Console()

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.5])
ViewerTypes = Literal["launcher", "video", "simple"]


def map_genotype_to_body(genome_data: dict[str, Any] | TreeGenome) -> mujoco.MjSpec | None:
    genome = TreeGenome.from_dict(genome_data) if isinstance(genome_data, dict) else genome_data
    try:
        robot_graph = genome.to_networkx()
        if robot_graph.number_of_nodes() == 0:
            return None
        return construct_mjspec_from_graph(robot_graph).spec
    except Exception:
        return None


def map_genotype_to_brain(cpg: SimpleCPG, full_genome: list[float]) -> None:
    n = cpg.phase.shape[0]
    params = np.array(full_genome)

    required_size = n * 5
    if required_size > len(params):
        params = np.resize(params, required_size)
    else:
        params = params[:required_size]

    p_phase = params[0 * n : 1 * n]
    p_w = params[1 * n : 2 * n]
    p_amp = params[2 * n : 3 * n]
    p_ha = params[3 * n : 4 * n]
    p_b = params[4 * n : 5 * n]

    with torch.no_grad():
        cpg.phase.data.copy_(torch.from_numpy(p_phase * np.pi).float())
        cpg.w.data.copy_(torch.from_numpy(0.2 + (3.8 * (p_w + 1.0) / 2.0)).float())
        cpg.amplitudes.data.copy_(torch.from_numpy(0.5 + (3.5 * (p_amp + 1.0) / 2.0)).float())
        cpg.ha.data.copy_(torch.from_numpy(p_ha * 2.0).float())
        cpg.b.data.copy_(torch.from_numpy(p_b * 0.5).float())


def load_best_individual_from_db(db_path: Path) -> Individual | None:
    if not db_path.exists():
        console.log(f"[red]Database not found:[/red] {db_path}")
        return None

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        statement = (
            select(Individual)
            .where(Individual.requires_eval == False)  # noqa: E712
            .where(Individual.fitness_ != None)  # noqa: E711
            .order_by(col(Individual.fitness_).asc())
        )
        return session.exec(statement).first()


def run_simulation(mode: ViewerTypes, ind: Individual, duration: float, out_dir: Path) -> float:
    mujoco.set_mjcb_control(None)

    genotype = cast(dict[str, Any], ind.genotype)

    spec = map_genotype_to_body(genotype["morph"])
    if spec is None:
        return float("inf")

    try:
        model = spec.compile()
    except Exception:
        return float("inf")

    if model.nu == 0:
        return float("inf")

    world = SimpleFlatWorldWithTarget()
    world.spawn(spec, position=SPAWN_POSITION)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    adj_dict = create_fully_connected_adjacency(model.nu)
    cpg = SimpleCPG(adj_dict)
    map_genotype_to_brain(cpg, cast(list[float], genotype["ctrl"]))

    tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
    ctrl = Controller(
        controller_callback_function=lambda m, d, *a, **k: cpg.forward(d.time),
        # Match EA sampling frequency for consistent fitness calculation.
        time_steps_per_save=20,
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)
    mujoco.set_mjcb_control(lambda m, d: ctrl.set_control(m, d, duration=duration))
    
    # Set seed for reproducible physics simulation.
    sim_seed = ind.tags.get("seed", hash(ind.id) % (2**31))
    np.random.seed(sim_seed)
    mujoco.mj_resetData(model, data)

    if mode == "simple":
        steps_required = int(duration / model.opt.timestep)
        for _ in range(steps_required):
            mujoco.mj_step(model, data)
    elif mode == "video":
        out_dir.mkdir(exist_ok=True, parents=True)
        recorder = VideoRecorder(output_folder=str(out_dir), file_name=f"db_best_{ind.id}")
        video_renderer(model, data, duration=duration, video_recorder=recorder)
    elif mode == "launcher":
        viewer.launch(model=model, data=data)

    if not tracker.history["xpos"]:
        return float("inf")

    first_key = list(tracker.history["xpos"].keys())[0]
    traj = tracker.history["xpos"][first_key]
    if not traj:
        return float("inf")

    # Use the same fitness calculation as the EA for consistency.
    # Ignore first second to let the robot settle/fall before scoring gait.
    delay_time = min(1.0, duration)
    sample_dt = model.opt.timestep * ctrl.time_steps_per_save
    if sample_dt <= 0:
        start_idx = 0
    else:
        start_idx = int(np.ceil(delay_time / sample_dt))
    start_idx = min(max(0, start_idx), max(0, len(traj) - 1))
    
    # Evaluate sustained locomotion, not just final position.
    # Split post-delay period in half to measure consistency.
    mid_idx = start_idx + (len(traj) - start_idx) // 2
    mid_idx = min(max(start_idx + 1, mid_idx), len(traj) - 1)
    
    pos_start = np.array(traj[start_idx])
    pos_mid = np.array(traj[mid_idx])
    pos_final = np.array(traj[-1])
    
    # Target direction from starting position after settling.
    to_target_xy = TARGET_POSITION[:2] - pos_start[:2]
    target_norm = float(np.linalg.norm(to_target_xy))
    if target_norm < 1e-8:
        target_dir = np.array([1.0, 0.0])
    else:
        target_dir = to_target_xy / target_norm
    
    # Measure progress in both halves of the active period.
    move1_xy = pos_mid[:2] - pos_start[:2]
    move2_xy = pos_final[:2] - pos_mid[:2]
    
    progress1 = float(np.dot(move1_xy, target_dir))
    progress2 = float(np.dot(move2_xy, target_dir))
    
    # Penalize robots that move in first half but stop/reverse in second half.
    sustained_progress = min(progress1, progress2)
    total_progress = progress1 + progress2
    
    # Lateral deviation from straight line to target.
    total_move_xy = pos_final[:2] - pos_start[:2]
    total_forward = float(np.dot(total_move_xy, target_dir))
    lateral_drift = float(np.linalg.norm(total_move_xy - total_forward * target_dir))
    
    # Distance to target at end.
    dist_final = float(np.linalg.norm(pos_final[:2] - TARGET_POSITION[:2]))
    
    # Fitness (minimize): prioritize sustained forward gait over final position.
    fitness = dist_final - 1.0 * total_progress - 1.5 * max(0.0, sustained_progress) + 0.3 * lateral_drift
    return fitness



def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate best individual from DB")
    parser.add_argument("--db_path", type=str, required=True, help="Path to database.db")
    parser.add_argument(
        "--viewer",
        type=str,
        default="launcher",
        choices=["simple", "video", "launcher"],
        help="Simulation viewer mode",
    )
    parser.add_argument("--dur", type=float, default=20.0, help="Simulation duration")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    best = load_best_individual_from_db(db_path)
    if best is None:
        console.log("[red]No evaluated individual found in DB.[/red]")
        return

    console.log(f"Loaded best from DB: id={best.id}, fitness={best.fitness_}")
    replayed_fitness = run_simulation(args.viewer, best, duration=args.dur, out_dir=db_path.parent / "videos")
    console.log(f"Replayed fitness: {replayed_fitness:.4f} (DB stored: {best.fitness_:.4f})")



if __name__ == "__main__":
    main()
