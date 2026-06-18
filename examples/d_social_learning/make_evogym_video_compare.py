"""Render best-policy videos for both brain architectures on the EvoGym walker.

Reads best θ from __data__/results_evogym/{distributed,standard}_best_theta.npy
and writes evogym_{distributed,standard}.mp4 to __data__/vids/.

# Environment: evogym-venv (Python 3.10) — EvoGym requires Python 3.10.
#              Do NOT run with the main uv/ariel venv.

Usage:
    evogym-venv/bin/python examples/d_social_learning/make_evogym_video_compare.py [--results-dir __data__/results_evogym] [--out-dir __data__/vids] [--steps 500]
"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import imageio
import gymnasium as gym
from evogym import EvoWorld, utils
from evogym.envs import EvoGymBase
from rich.console import Console

from ariel.simulation.controllers import DistributedMLP, StandardMLP
from evogym_adapter import (
    body_to_adjacency,
    get_actuator_order,
    get_node_inputs,
    get_standard_obs,
    scale_actions,
)

console = Console()

_WORLD_JSON = os.path.join(os.path.dirname(__file__), "../../evogymtest/world_data/simple_environment.json")

SAMPLE_BODY = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=float)

FPS = 50


def make_env(body):
    world = EvoWorld.from_json(_WORLD_JSON)
    world.add_from_array("robot", body, 1, 1, connections=utils.get_full_connectivity(body))

    class _Env(EvoGymBase):
        def __init__(self):
            super().__init__(world, render_mode="rgb_array")
            n_act = self.get_actuator_indices("robot").size
            self.action_space = gym.spaces.Box(low=0.6, high=1.6, shape=(n_act,), dtype=np.float64)

        def step(self, action):
            super().step({"robot": action})
            return None, 0.0, False, False, {}

        def reset(self, **kwargs):
            super().reset()
            return None, {}

    return _Env()


def render_video(brain_type: str, theta: np.ndarray, out_path: str, n_steps: int) -> None:
    body = SAMPLE_BODY
    adjacency = body_to_adjacency(body)
    nu = len(get_actuator_order(body))

    if brain_type == "distributed":
        brain = DistributedMLP(n_neighbors=8)
    else:
        brain = StandardMLP(n_actuators=nu)
    brain.set_theta(theta)

    env = make_env(body)
    env.reset()

    frames = []
    console.log(f"Rendering {brain_type} EvoGym for {n_steps} steps...")

    for step in range(n_steps):
        if brain_type == "distributed":
            node_inputs, t = get_node_inputs(env.sim, body, adjacency, step)
            raw = brain.forward_all(node_inputs, t)
        else:
            obs = get_standard_obs(env.sim, body, step)
            raw = brain.forward(obs)
        env.step(scale_actions(raw))
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    env.close()

    if not frames:
        console.log(f"[red]No frames for {brain_type}[/red]")
        return

    console.log(f"  {len(frames)} frames, writing {out_path}...")
    imageio.mimsave(out_path, frames, fps=FPS)
    console.log(f"[green]Saved → {out_path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="__data__/results_evogym")
    parser.add_argument("--out-dir", default="__data__/vids")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for brain_type in ("distributed", "standard"):
        theta_path = os.path.join(args.results_dir, f"{brain_type}_best_theta.npy")
        if not os.path.exists(theta_path):
            console.log(f"[yellow]Skipping {brain_type}: {theta_path} not found[/yellow]")
            continue
        theta = np.load(theta_path)
        out_path = os.path.join(args.out_dir, f"evogym_{brain_type}.mp4")
        render_video(brain_type, theta, out_path, args.steps)


if __name__ == "__main__":
    main()
