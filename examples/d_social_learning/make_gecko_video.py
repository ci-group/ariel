"""Render a video of the gecko playing back a learned θ vector."""

from __future__ import annotations

import argparse
import os

import imageio
import mujoco
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers import DistributedMLP
from ariel.simulation.environments import SimpleFlatWorld

from morphology_adapter import gecko_graph, MorphologyAdapter
from gecko_utils import scale_actions

console = Console()

SPAWN_POS = (-0.8, 0.0, 0.1)
FPS = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", default="__data__/best_theta_gecko.npy")
    parser.add_argument("--out", default="__data__/vids/gecko_gait.mp4")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    theta = np.load(args.theta)
    console.log(f"Loaded θ from {args.theta}, size={len(theta)}")

    graph = gecko_graph()
    adapter = MorphologyAdapter.from_graph(graph)
    core = construct_mjspec_from_graph(graph)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    brain = DistributedMLP(n_neighbors=6)
    brain.set_theta(theta)

    renderer = mujoco.Renderer(model, height=480, width=640)

    CTRL_EVERY = 100
    frames = []
    ctrl_step = 0
    sim_step = 0
    dt = model.opt.timestep
    render_every = max(1, int(round(1.0 / (FPS * dt))))

    console.log(f"Rendering {args.duration}s at {FPS}fps (every {render_every} sim steps)...")

    while data.time < args.duration:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            data.ctrl[:] = scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera="pretty-cam")
            frames.append(renderer.render())
        sim_step += 1

    renderer.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    console.log(f"Captured {len(frames)} frames, writing {args.out}...")
    imageio.mimsave(args.out, frames, fps=FPS)
    console.log(f"[green]Saved → {args.out}[/green]")


if __name__ == "__main__":
    main()
