"""ARIEL gecko architecture comparison experiment.

Runs 10 repetitions of CMA-ES for both DistributedMLP and StandardMLP on the
gecko morphology. Saves per-generation best fitness histories and the overall
best θ for each architecture.

Usage:
    uv run examples/d_social_learning/experiment_gecko.py [--gens 50] [--pop 16] [--reps 10] [--out-dir __data__/results_gecko]
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers import CMAESLearner, DistributedMLP, StandardMLP
from ariel.simulation.environments import SimpleFlatWorld

from morphology_adapter import gecko_graph, MorphologyAdapter
from gecko_utils import scale_actions, get_standard_obs

console = Console()

DURATION = 30.0
SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 100

_GRAPH = gecko_graph()
ADAPTER = MorphologyAdapter.from_graph(_GRAPH)


def build_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core = construct_mjspec_from_graph(_GRAPH)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def evaluate_distributed(model, data, brain: DistributedMLP, theta: np.ndarray) -> float:
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)
    ctrl_step = sim_step = 0
    while data.time < DURATION:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = ADAPTER.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            data.ctrl[:] = scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        sim_step += 1
    return float(data.qpos[0])


def evaluate_standard(model, data, brain: StandardMLP, theta: np.ndarray) -> float:
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)
    ctrl_step = sim_step = 0
    while data.time < DURATION:
        if sim_step % CTRL_EVERY == 0:
            obs = get_standard_obs(model, data, ADAPTER.actuator_to_module, ADAPTER._joint_name_to_module)
            raw = brain.forward(obs)
            data.ctrl[:] = scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        sim_step += 1
    return float(data.qpos[0])


def run_one(brain_type: str, gens: int, pop: int) -> tuple[list[float], np.ndarray]:
    """Run one repetition. Returns (gen_best_history, best_theta)."""
    model, data = build_world()
    nu = len(ADAPTER.actuator_to_module)

    if brain_type == "distributed":
        brain = DistributedMLP(n_neighbors=6)
        learner = CMAESLearner(n_params=brain.n_params, pop_size=pop)
        eval_fn = lambda theta: evaluate_distributed(model, data, brain, theta)
    else:
        brain = StandardMLP(n_actuators=nu)
        learner = CMAESLearner(n_params=brain.n_params, pop_size=pop)
        eval_fn = lambda theta: evaluate_standard(model, data, brain, theta)

    gen_bests: list[float] = []
    for _ in range(gens):
        candidates = learner.ask()
        fitnesses = [eval_fn(theta) for theta in candidates]
        learner.tell(candidates, fitnesses)
        gen_bests.append(learner.best_fitness)

    return gen_bests, learner.best_theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=50)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out-dir", default="__data__/results_gecko")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    console.rule("[bold cyan]ARIEL Gecko Architecture Comparison[/bold cyan]")
    console.log(f"gens={args.gens}, pop={args.pop}, reps={args.reps}, nu={len(ADAPTER.actuator_to_module)}")

    for brain_type in ("distributed", "standard"):
        console.rule(f"[bold yellow]{brain_type.upper()}[/bold yellow]")
        all_histories: list[list[float]] = []
        best_of_best_fitness = -float("inf")
        best_of_best_theta: np.ndarray | None = None

        for rep in range(args.reps):
            console.log(f"  rep {rep+1}/{args.reps}...")
            gen_bests, theta = run_one(brain_type, args.gens, args.pop)
            all_histories.append(gen_bests)
            if gen_bests[-1] > best_of_best_fitness:
                best_of_best_fitness = gen_bests[-1]
                best_of_best_theta = theta
            console.log(f"    final best = {gen_bests[-1]:.4f}m")

        histories_arr = np.array(all_histories)  # (reps, gens)
        np.save(os.path.join(args.out_dir, f"{brain_type}_histories.npy"), histories_arr)
        np.save(os.path.join(args.out_dir, f"{brain_type}_best_theta.npy"), best_of_best_theta)
        console.log(f"[green]Saved {brain_type} results → {args.out_dir}/[/green]")
        console.log(f"[green]Best x-displacement across reps: {best_of_best_fitness:.4f}m[/green]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
