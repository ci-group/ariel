"""ARIEL gecko distributed-brain learning script.

Learns a DistributedMLP brain for the fixed gecko morphology using CMA-ES,
then saves the best weights and replays the learned gait.

Usage:
    uv run examples/d_social_learning/ariel_gecko_learn.py [--gens N] [--pop N]
"""

from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers import CMAESLearner, DistributedMLP
from ariel.simulation.environments import SimpleFlatWorld

from morphology_adapter import gecko_graph, MorphologyAdapter
from gecko_utils import scale_actions

console = Console()

DURATION = 30.0       # seconds of simulation per episode
SPAWN_POS = (-0.8, 0.0, 0.1)

_GRAPH = gecko_graph()
ADAPTER = MorphologyAdapter.from_graph(_GRAPH)


def build_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core = construct_mjspec_from_graph(_GRAPH)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def evaluate(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    brain: DistributedMLP,
    theta: np.ndarray,
) -> float:
    CTRL_EVERY = 100  # call brain every 100 sim steps (0.2s at 500Hz)
    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)
    ctrl_step = 0
    sim_step = 0
    while data.time < DURATION:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = ADAPTER.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            data.ctrl[:] = scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        sim_step += 1
    return float(data.qpos[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=20, help="CMA-ES generations")
    parser.add_argument("--pop", type=int, default=16, help="CMA-ES population size")
    parser.add_argument("--no-render", action="store_true", help="Skip replay viewer")
    parser.add_argument("--out", default="__data__/best_theta_gecko.npy", help="Output .npy path")
    args = parser.parse_args()

    console.rule("[bold cyan]ARIEL Gecko Distributed-Brain CMA-ES Learning[/bold cyan]")
    console.log(
        f"gens={args.gens}, pop={args.pop}, duration={DURATION}s, "
        f"actuators={len(ADAPTER.actuator_to_module)}"
    )

    model, data = build_world()
    brain = DistributedMLP(n_neighbors=6)
    learner = CMAESLearner(n_params=brain.n_params, pop_size=args.pop)

    console.log(f"θ size = {brain.n_params}, MuJoCo actuators = {model.nu}")

    for gen in range(args.gens):
        candidates = learner.ask()
        fitnesses = [evaluate(model, data, brain, theta) for theta in candidates]
        learner.tell(candidates, fitnesses)
        console.log(
            f"gen {gen+1:3d}/{args.gens}  "
            f"best={learner.best_fitness:.4f}m  "
            f"gen_best={max(fitnesses):.4f}m  "
            f"gen_mean={np.mean(fitnesses):.4f}m"
        )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, learner.best_theta)
    console.log(f"[green]Saved best θ → {out_path}[/green]")
    console.log(f"[green]Best x-displacement: {learner.best_fitness:.4f}m[/green]")

    if args.no_render:
        return

    console.rule("[bold green]Replaying best gait[/bold green]")
    model, data = build_world()
    brain.set_theta(learner.best_theta)
    mujoco.mj_resetData(model, data)

    if sys.platform == "darwin":
        # macOS: use active viewer (blocks until window is closed)
        console.log("[yellow]Active MuJoCo viewer — close the window to exit.[/yellow]")

        step_counter = [0]

        def ctrl_cb(m, d):
            node_inputs, t = ADAPTER.get_node_inputs(m, d, step_counter[0])
            raw = brain.forward_all(node_inputs, t)
            d.ctrl[:] = scale_actions(raw)
            step_counter[0] += 1

        mujoco.set_mjcb_control(ctrl_cb)
        mujoco.viewer.launch(model=model, data=data)
        mujoco.set_mjcb_control(None)
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            step = 0
            sim_start = time.time()
            replay_duration = DURATION * 3
            while v.is_running() and (time.time() - sim_start) < replay_duration:
                t_step = time.time()
                node_inputs, t = ADAPTER.get_node_inputs(model, data, step)
                raw = brain.forward_all(node_inputs, t)
                data.ctrl[:] = scale_actions(raw)
                mujoco.mj_step(model, data)
                v.sync()
                step += 1
                remaining = model.opt.timestep - (time.time() - t_step)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()
