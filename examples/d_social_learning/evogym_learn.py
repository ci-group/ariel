"""EvoGym distributed-brain learning script.

Learns a DistributedMLP brain for a fixed 5x5 walker morphology using CMA-ES,
then saves the best weights and replays the learned gait.

# Environment: evogym-venv (Python 3.10) — EvoGym requires Python 3.10.
#              Do NOT run with the main uv/ariel venv.

Usage:
    evogym-venv/bin/python examples/d_social_learning/evogym_learn.py [--gens N] [--pop N] [--steps N] [--no-render]
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import gymnasium as gym
from evogym import EvoWorld, utils
from evogym.envs import EvoGymBase
from rich.console import Console

from ariel.simulation.controllers import CMAESLearner, DistributedMLP
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


def make_env(body: np.ndarray) -> gym.Env:
    world = EvoWorld.from_json(_WORLD_JSON)
    world.add_from_array("robot", body, 1, 1, connections=utils.get_full_connectivity(body))

    class _Env(EvoGymBase):
        def __init__(self):
            super().__init__(world)
            n_act = self.get_actuator_indices("robot").size
            self.action_space = gym.spaces.Box(low=0.6, high=1.6, shape=(n_act,), dtype=np.float64)

        def step(self, action):
            pos1 = self.object_pos_at_time(self.get_time(), "robot")
            super().step({"robot": action})
            pos2 = self.object_pos_at_time(self.get_time(), "robot")
            reward = float(np.mean(pos2[0]) - np.mean(pos1[0]))
            return None, reward, False, False, {}

        def reset(self, **kwargs):
            super().reset()
            return None, {}

    return _Env()


def evaluate(env, brain: DistributedMLP, theta: np.ndarray, body: np.ndarray, adjacency: dict, n_steps: int) -> float:
    brain.set_theta(theta)
    env.reset()
    total_reward = 0.0
    for step in range(n_steps):
        node_inputs, t = get_node_inputs(env.sim, body, adjacency, step)
        raw = brain.forward_all(node_inputs, t)
        _, reward, done, _, _ = env.step(scale_actions(raw))
        total_reward += reward
        if done:
            break
    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=20)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out", default="__data__/best_theta_evogym.npy")
    args = parser.parse_args()

    console.rule("[bold cyan]EvoGym Distributed-Brain CMA-ES Learning[/bold cyan]")
    console.log(f"gens={args.gens}, pop={args.pop}, steps={args.steps}, actuators={len(get_actuator_order(SAMPLE_BODY))}")

    body = SAMPLE_BODY
    adjacency = body_to_adjacency(body)
    env = make_env(body)
    brain = DistributedMLP(n_neighbors=8)
    learner = CMAESLearner(n_params=brain.n_params, pop_size=args.pop)
    console.log(f"θ size = {brain.n_params}")

    for gen in range(args.gens):
        candidates = learner.ask()
        fitnesses = [evaluate(env, brain, theta, body, adjacency, args.steps) for theta in candidates]
        learner.tell(candidates, fitnesses)
        console.log(f"gen {gen+1:3d}/{args.gens}  best={learner.best_fitness:.4f}  gen_best={max(fitnesses):.4f}  gen_mean={np.mean(fitnesses):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, learner.best_theta)
    console.log(f"[green]Saved best θ → {args.out}[/green]")
    console.log(f"[green]Best fitness: {learner.best_fitness:.4f}[/green]")

    if args.no_render:
        env.close()
        return

    console.rule("[bold green]Replaying best gait[/bold green]")
    env_render = make_env(body)
    brain.set_theta(learner.best_theta)
    env_render.reset()

    try:
        import imageio
        frames = []
        for step in range(args.steps):
            node_inputs, t = get_node_inputs(env_render.sim, body, adjacency, step)
            raw = brain.forward_all(node_inputs, t)
            env_render.step(scale_actions(raw))
            frame = env_render.render()
            if frame is not None:
                frames.append(frame)
        if frames:
            gif_path = args.out.replace(".npy", "_replay.gif")
            imageio.mimsave(gif_path, frames, fps=25)
            console.log(f"[green]Saved replay → {gif_path}[/green]")
    except Exception as exc:
        console.log(f"[yellow]Replay skipped: {exc}[/yellow]")
    finally:
        env_render.close()

    env.close()


if __name__ == "__main__":
    main()
