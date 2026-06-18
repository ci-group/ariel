"""EvoGym architecture comparison experiment.

Runs N repetitions of CMA-ES for both DistributedMLP and StandardMLP on the
fixed 5×5 EvoGym walker. Saves per-generation best fitness histories and the
overall best θ for each architecture.

# Environment: evogym-venv (Python 3.10) — EvoGym requires Python 3.10.
#              Do NOT run with the main uv/ariel venv.

Usage:
    evogym-venv/bin/python examples/d_social_learning/experiment_evogym.py [--gens 50] [--pop 16] [--reps 10] [--out-dir __data__/results_evogym]
"""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
from evogym import EvoWorld, utils
from evogym.envs import EvoGymBase
from rich.console import Console

from ariel.simulation.controllers import CMAESLearner, DistributedMLP, StandardMLP
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


def evaluate_distributed(env, brain: DistributedMLP, theta: np.ndarray, body, adjacency, n_steps: int) -> float:
    brain.set_theta(theta)
    env.reset()
    total = 0.0
    for step in range(n_steps):
        node_inputs, t = get_node_inputs(env.sim, body, adjacency, step)
        raw = brain.forward_all(node_inputs, t)
        _, reward, done, _, _ = env.step(scale_actions(raw))
        total += reward
        if done:
            break
    return total


def evaluate_standard(env, brain: StandardMLP, theta: np.ndarray, body, n_steps: int) -> float:
    brain.set_theta(theta)
    env.reset()
    total = 0.0
    for step in range(n_steps):
        obs = get_standard_obs(env.sim, body, step)
        raw = brain.forward(obs)
        _, reward, done, _, _ = env.step(scale_actions(raw))
        total += reward
        if done:
            break
    return total


def run_one(brain_type: str, body: np.ndarray, adjacency: dict, gens: int, pop: int, n_steps: int) -> tuple[list[float], np.ndarray]:
    env = make_env(body)
    nu = len(get_actuator_order(body))

    if brain_type == "distributed":
        brain = DistributedMLP(n_neighbors=8)
        learner = CMAESLearner(n_params=brain.n_params, pop_size=pop)
        eval_fn = lambda theta: evaluate_distributed(env, brain, theta, body, adjacency, n_steps)
    else:
        brain = StandardMLP(n_actuators=nu)
        learner = CMAESLearner(n_params=brain.n_params, pop_size=pop)
        eval_fn = lambda theta: evaluate_standard(env, brain, theta, body, n_steps)

    gen_bests: list[float] = []
    for _ in range(gens):
        candidates = learner.ask()
        fitnesses = [eval_fn(theta) for theta in candidates]
        learner.tell(candidates, fitnesses)
        gen_bests.append(learner.best_fitness)

    env.close()
    return gen_bests, learner.best_theta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=50)
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out-dir", default="__data__/results_evogym")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    body = SAMPLE_BODY
    adjacency = body_to_adjacency(body)
    nu = len(get_actuator_order(body))

    console.rule("[bold cyan]EvoGym Architecture Comparison[/bold cyan]")
    console.log(f"gens={args.gens}, pop={args.pop}, steps={args.steps}, reps={args.reps}, actuators={nu}")

    for brain_type in ("distributed", "standard"):
        console.rule(f"[bold yellow]{brain_type.upper()}[/bold yellow]")
        all_histories: list[list[float]] = []
        best_of_best_fitness = -float("inf")
        best_of_best_theta: np.ndarray | None = None

        for rep in range(args.reps):
            console.log(f"  rep {rep+1}/{args.reps}...")
            gen_bests, theta = run_one(brain_type, body, adjacency, args.gens, args.pop, args.steps)
            all_histories.append(gen_bests)
            if gen_bests[-1] > best_of_best_fitness:
                best_of_best_fitness = gen_bests[-1]
                best_of_best_theta = theta
            console.log(f"    final best = {gen_bests[-1]:.4f}")

        histories_arr = np.array(all_histories)
        np.save(os.path.join(args.out_dir, f"{brain_type}_histories.npy"), histories_arr)
        np.save(os.path.join(args.out_dir, f"{brain_type}_best_theta.npy"), best_of_best_theta)
        console.log(f"[green]Saved {brain_type} results → {args.out_dir}/[/green]")
        console.log(f"[green]Best fitness across reps: {best_of_best_fitness:.4f}[/green]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
