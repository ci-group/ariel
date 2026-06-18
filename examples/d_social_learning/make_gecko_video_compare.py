"""Render best-policy videos for both brain architectures on the ARIEL gecko.

Reads best θ from __data__/results_gecko/{distributed,standard}_best_theta.npy and
writes gecko_{distributed,standard}.mp4 to __data__/vids/.

Usage:
    uv run examples/d_social_learning/make_gecko_video_compare.py [--results-dir __data__/results_gecko] [--out-dir __data__/vids] [--duration 8]
"""

from __future__ import annotations

import argparse
import os

import imageio
import mujoco
import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers import DistributedMLP, StandardMLP
from ariel.simulation.environments import SimpleFlatWorld

from morphology_adapter import gecko_graph, MorphologyAdapter
from gecko_utils import scale_actions, get_standard_obs

console = Console()

SPAWN_POS = (-0.8, 0.0, 0.1)
FPS = 50
CTRL_EVERY = 100


_GRAPH = gecko_graph()
ADAPTER = MorphologyAdapter.from_graph(_GRAPH)


def build_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core = construct_mjspec_from_graph(_GRAPH)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data


def render_video(
    brain_type: str,
    theta: np.ndarray,
    out_path: str,
    duration: float,
) -> None:
    model, data = build_world()
    nu = len(ADAPTER.actuator_to_module)
    if brain_type == "distributed":
        brain = DistributedMLP(n_neighbors=6)
    else:
        brain = StandardMLP(n_actuators=nu)
    brain.set_theta(theta)

    renderer = mujoco.Renderer(model, height=480, width=640)
    dt = model.opt.timestep
    render_every = max(1, int(round(1.0 / (FPS * dt))))

    frames = []
    ctrl_step = sim_step = 0

    console.log(f"Rendering {brain_type} gecko for {duration}s...")

    while data.time < duration:
        if sim_step % CTRL_EVERY == 0:
            if brain_type == "distributed":
                node_inputs, t = ADAPTER.get_node_inputs(model, data, ctrl_step)
                raw = brain.forward_all(node_inputs, t)
            else:
                obs = get_standard_obs(model, data, ADAPTER.actuator_to_module, ADAPTER._joint_name_to_module)
                raw = brain.forward(obs)
            data.ctrl[:] = scale_actions(raw)
            ctrl_step += 1
        mujoco.mj_step(model, data)
        if sim_step % render_every == 0:
            renderer.update_scene(data, camera="pretty-cam")
            frames.append(renderer.render())
        sim_step += 1

    renderer.close()
    console.log(f"  {len(frames)} frames, writing {out_path}...")
    imageio.mimsave(out_path, frames, fps=FPS)
    console.log(f"[green]Saved → {out_path}[/green]")
    final_x = float(data.qpos[0])
    console.log(f"  final x-displacement: {final_x:.4f}m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="__data__/results_gecko")
    parser.add_argument("--out-dir", default="__data__/vids")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for brain_type in ("distributed", "standard"):
        theta_path = os.path.join(args.results_dir, f"{brain_type}_best_theta.npy")
        if not os.path.exists(theta_path):
            console.log(f"[yellow]Skipping {brain_type}: {theta_path} not found[/yellow]")
            continue
        theta = np.load(theta_path)
        out_path = os.path.join(args.out_dir, f"gecko_{brain_type}.mp4")
        render_video(brain_type, theta, out_path, args.duration)


if __name__ == "__main__":
    main()
