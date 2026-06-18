"""Render a video of the EvoGym walker playing back a learned θ vector.

# Environment: evogym-venv (Python 3.10) — EvoGym requires Python 3.10.
#              Do NOT run with the main uv/ariel venv.

Usage:
    evogym-venv/bin/python examples/d_social_learning/make_evogym_video.py [--theta __data__/best_theta_evogym.npy] [--out __data__/vids/evogym_gait.mp4]
"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import imageio
from evogym import EvoWorld, utils
from evogym.envs import EvoGymBase
import gymnasium as gym
from rich.console import Console

from ariel.simulation.controllers import DistributedMLP
from evogym_adapter import body_to_adjacency, get_actuator_order, get_node_inputs, scale_actions

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", default="__data__/best_theta_evogym.npy")
    parser.add_argument("--out", default="__data__/vids/evogym_gait.mp4")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    theta = np.load(args.theta)
    console.log(f"Loaded θ from {args.theta}, size={len(theta)}")

    body = SAMPLE_BODY
    adjacency = body_to_adjacency(body)
    env = make_env(body)
    env.reset()

    brain = DistributedMLP(n_neighbors=8)
    brain.set_theta(theta)

    frames = []
    console.log(f"Rendering {args.steps} steps at {FPS}fps...")

    for step in range(args.steps):
        node_inputs, t = get_node_inputs(env.sim, body, adjacency, step)
        raw = brain.forward_all(node_inputs, t)
        env.step(scale_actions(raw))
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    env.close()

    if not frames:
        console.log("[red]No frames captured.[/red]")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    console.log(f"Captured {len(frames)} frames, writing {args.out}...")
    imageio.mimsave(args.out, frames, fps=FPS)
    console.log(f"[green]Saved → {args.out}[/green]")


if __name__ == "__main__":
    main()
