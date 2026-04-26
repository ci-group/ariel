"""
Fast-converging tree morphology + controller co-evolution example.

Key ideas:
- Curriculum on simulation duration (short -> long)
- Stronger parent selection pressure
- Controller-first mutation bias early
- Progress-shaped fitness
- Occasional novelty injection
"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install

install()
console = Console()

from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ALLOWED_ROTATIONS,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    get_tree_depth,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
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


parser = argparse.ArgumentParser(description="Fast Joint Evolution: Body + Brain (trees)")
parser.add_argument("--budget", type=int, default=20, help="Number of generations")
parser.add_argument("--pop", type=int, default=30, help="Population size")
parser.add_argument("--dur", type=int, default=20, help="Max simulation duration")
parser.add_argument(
    "--final-viewer",
    type=str,
    default="none",
    choices=["none", "simple", "video", "launcher"],
    help="Viewer mode for final best individual",
)
args = parser.parse_args()

DURATION: int = args.dur
POP_SIZE: int = args.pop
BUDGET: int = args.budget
NUM_MODULES: int = 10
CTRL_GENOME_SIZE: int = NUM_MODULES * 5

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.5])

Population = list[Individual]
ViewerTypes = Literal["launcher", "video", "simple"]

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


class Evolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_of_generations=BUDGET,
            target_population_size=POP_SIZE,
            db_file_path=DATA / "database.db",
        )
        self.generation = 0
        self.best_avg_fitness_seen = float("inf")
        self.stagnation_generations = 0

    def get_eval_duration(self) -> int:
        if self.generation < 10:
            return min(5, DURATION)
        if self.generation < 25:
            return min(10, DURATION)
        return DURATION

    def map_genotype_to_body(self, genome_data: dict | TreeGenome) -> mujoco.MjSpec | None:
        genome = TreeGenome.from_dict(genome_data) if isinstance(genome_data, dict) else genome_data
        try:
            robot_graph = genome.to_networkx()
            if robot_graph.number_of_nodes() == 0:
                return None
            return construct_mjspec_from_graph(robot_graph).spec
        except Exception:
            return None

    def map_genotype_to_brain(self, cpg: SimpleCPG, full_genome: list[float]) -> None:
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

    def get_joint_count(self, genome: TreeGenome) -> int:
        spec = self.map_genotype_to_body(genome)
        if spec is None:
            return 0
        try:
            return spec.compile().nu
        except Exception:
            return 0

    def init_ctrl_prior(self) -> list[float]:
        n = NUM_MODULES
        phase = np.linspace(-0.5, 0.5, n)
        w = np.full(n, 0.0)
        amp = np.full(n, 0.2)
        ha = np.zeros(n)
        b = np.zeros(n)
        base = np.concatenate([phase, w, amp, ha, b])
        noise = RNG.normal(0.0, 0.15, size=base.shape)
        return np.clip(base + noise, -1.0, 1.0).tolist()

    def mutate_ctrl_vector(self, genome: list[float]) -> list[float]:
        arr = np.array(genome)
        mask = RNG.random(arr.shape) < 0.35
        noise = RNG.normal(0, 0.4, arr.shape)
        arr[mask] += noise[mask]
        return np.clip(arr, -1.0, 1.0).tolist()

    def crossover_morphologies(self, parent1: Individual, parent2: Individual) -> TreeGenome:
        t1 = parent1.genotype["morph"]
        t2 = parent2.genotype["morph"]
        if isinstance(t1, dict):
            t1 = TreeGenome.from_dict(t1)
        if isinstance(t2, dict):
            t2 = TreeGenome.from_dict(t2)
        child1, child2 = crossover_subtree(t1, t2)
        return child1 if RNG.random() < 0.5 else child2

    def mutate_morphology(self, genome: TreeGenome) -> TreeGenome:
        new = copy.deepcopy(genome)

        if self.generation < 15:
            probs = [0.8, 0.15, 0.03, 0.02]
        else:
            probs = [0.65, 0.25, 0.06, 0.04]

        # If search stagnates, increase structural exploration.
        if self.stagnation_generations >= 5:
            probs = [0.45, 0.38, 0.09, 0.08]

        mutation_type = RNG.choice(["point", "subtree", "shrink", "hoist"], p=probs)

        if mutation_type == "point":
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            mutate_subtree_replacement(new, max_modules=NUM_MODULES)
        elif mutation_type == "shrink":
            mutate_shrink(new)
        else:
            mutate_hoist(new)

        if RNG.random() < 0.3:
            noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
            if noncore:
                nid = random.choice(noncore)
                mtype = ModuleType[new.nodes[nid]["type"]]
                rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
                if rots:
                    new.nodes[nid]["rotation"] = random.choice(rots)

        _prune_invalid_edges(new)
        try:
            validate_genome_dict(new.to_dict())
            return new
        except ValueError:
            # Keep evolution stable: if mutation produces invalid morphology,
            # return the original genome instead of passing invalid structures.
            return copy.deepcopy(genome)

    def crossover_ctrl_vectors(self, ctrl1: list[float], ctrl2: list[float]) -> list[float]:
        arr1 = np.array(ctrl1)
        arr2 = np.array(ctrl2)
        size = max(len(arr1), len(arr2))
        if len(arr1) < size:
            arr1 = np.resize(arr1, size)
        if len(arr2) < size:
            arr2 = np.resize(arr2, size)
        mask = RNG.random(size) < 0.5
        child = np.where(mask, arr1, arr2)
        return child.tolist()

    def create_individual(self) -> Individual:
        # Keep initialization cheap for large populations.
        # random_tree() is typically valid and jointed in this domain.
        genome = random_tree(NUM_MODULES)
        ind = Individual()
        ind.genotype = {
            "morph": genome.to_dict(),
            "ctrl": self.init_ctrl_prior(),
        }
        ind.tags["ps"] = False
        ind.tags["valid"] = True
        ind.tags["debug_joints"] = -1
        # Store a deterministic seed derived from individual ID for reproducible physics.
        ind.tags["seed"] = hash(ind.id) % (2**31)
        return ind

    def reproduction(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = population

        new_offspring: list[Individual] = []
        target_pool = self.config.target_population_size * 2
        offspring_counter = 0
        max_iterations = target_pool * 3  # Safety limit
        iterations = 0

        ctrl_mut_prob = 0.6 if self.generation < 10 else (0.45 if self.generation < 20 else 0.3)
        if self.stagnation_generations >= 2:
            ctrl_mut_prob = min(0.75, ctrl_mut_prob + 0.15)

        novelty_period = 5 if self.stagnation_generations < 2 else 2

        while len(population) + len(new_offspring) < target_pool and iterations < max_iterations:
            iterations += 1
            if offspring_counter > 0 and offspring_counter % novelty_period == 0:
                new_offspring.append(self.create_individual())
                offspring_counter += 1
                continue

            use_sexual = len(parents) >= 2 and RNG.random() < 0.7
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                c_morph = self.crossover_morphologies(p1, p2)
                c_ctrl = self.crossover_ctrl_vectors(p1.genotype["ctrl"], p2.genotype["ctrl"])
            else:
                parent = random.choice(parents)
                c_morph = TreeGenome.from_dict(parent.genotype["morph"])
                c_ctrl = parent.genotype["ctrl"].copy()

            c_morph = self.mutate_morphology(c_morph)

            # Guard against morphology bloat (prevents late-generation slowdowns).
            if len(c_morph.nodes) > NUM_MODULES:
                c_morph = random_tree(NUM_MODULES)
            if get_tree_depth(c_morph) > 14:
                c_morph = random_tree(NUM_MODULES)

            if RNG.random() < ctrl_mut_prob:
                c_ctrl = self.mutate_ctrl_vector(c_ctrl)

            ind = Individual()
            ind.genotype = {"morph": c_morph.to_dict(), "ctrl": c_ctrl}
            ind.tags["ps"] = False
            ind.tags["valid"] = True
            ind.tags["debug_joints"] = -1
            new_offspring.append(ind)
            offspring_counter += 1

        if iterations >= max_iterations:
            console.log(f"[yellow]Warning: Reproduction hit iteration limit. Generated {len(new_offspring)} offspring (target: {target_pool - len(population)})[/yellow]")
        
        population.extend(new_offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        to_eval = [
            ind for ind in population if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        if not to_eval:
            return population

        duration = self.get_eval_duration()
        for ind in track(to_eval, description=f"Evaluating (dur={duration})..."):
            fitness = self.run_simulation("simple", ind, duration=duration)
            ind.fitness = fitness
            ind.requires_eval = False
        return population

    def parent_selection(self, population: Population) -> Population:
        population.sort(key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"))
        cutoff = max(2, int(len(population) * 0.35))
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff
        ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
        console.log(f"[cyan]Parent Selection: {ps_count}/{len(population)}[/cyan]")
        return population

    def survivor_selection(self, population: Population) -> Population:
        population.sort(key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"))
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        avg_fitness = np.mean(
            [ind.fitness_ for ind in survivors if ind.fitness_ is not None and ind.fitness_ != float("inf")]
        )

        # Track stagnation on average fitness (minimization objective).
        if avg_fitness < (self.best_avg_fitness_seen - 1e-2):
            self.best_avg_fitness_seen = float(avg_fitness)
            self.stagnation_generations = 0
        else:
            self.stagnation_generations += 1

        console.log(
            f"[green]Gen {self.generation:02d} | Avg fitness = {avg_fitness:.4f} | Eval dur = {self.get_eval_duration()} | Stag = {self.stagnation_generations}[/green]"
        )
        self.generation += 1
        return population

    def run_simulation(self, mode: ViewerTypes, ind: Individual, duration: float) -> float:
        mujoco.set_mjcb_control(None)
        expected_joints = ind.tags.get("debug_joints", 0)

        spec = None
        model = None
        attempts = 3 if mode != "simple" else 1
        for _ in range(attempts):
            temp_spec = self.map_genotype_to_body(ind.genotype["morph"])
            if temp_spec:
                try:
                    temp_model = temp_spec.compile()
                    if mode == "simple" or expected_joints < 0 or temp_model.nu == expected_joints:
                        spec = temp_spec
                        model = temp_model
                        break
                except Exception:
                    pass

        if model is None:
            spec = self.map_genotype_to_body(ind.genotype["morph"])
            if spec:
                model = spec.compile()
            else:
                return float("inf")

        if mode == "simple":
            if model.nu == 0:
                ind.tags["debug_joints"] = 0
                return float("inf")
            ind.tags["debug_joints"] = model.nu

        world = SimpleFlatWorldWithTarget()
        world.spawn(spec, position=SPAWN_POSITION)
        model = world.spec.compile()
        data = mujoco.MjData(model)

        adj_dict = create_fully_connected_adjacency(model.nu)
        cpg = SimpleCPG(adj_dict)
        self.map_genotype_to_brain(cpg, ind.genotype["ctrl"])

        tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
        ctrl = Controller(
            controller_callback_function=lambda m, d, *a, **k: cpg.forward(d.time),
            # Sample tracker frequently enough so a 1-second delay is meaningful.
            # Default (500) can be too coarse for short episodes.
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
            recorder = VideoRecorder(output_folder=str(DATA / "videos"), file_name=f"fast_{ind.id}")
            video_renderer(model, data, duration=duration, video_recorder=recorder)
        elif mode == "launcher":
            viewer.launch(model=model, data=data)

        if not tracker.history["xpos"]:
            return float("inf")

        first_key = list(tracker.history["xpos"].keys())[0]
        traj = tracker.history["xpos"][first_key]
        if not traj:
            return float("inf")

        # Ignore first second to let the robot settle/fall before scoring gait.
        # Convert delay time to tracker samples using MuJoCo timing.
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
        # Use minimum to ensure both halves show forward progress.
        sustained_progress = min(progress1, progress2)
        total_progress = progress1 + progress2
        
        # Lateral deviation from straight line to target.
        total_move_xy = pos_final[:2] - pos_start[:2]
        total_forward = float(np.dot(total_move_xy, target_dir))
        lateral_drift = float(np.linalg.norm(total_move_xy - total_forward * target_dir))
        
        # Distance to target at end (still matters, but less than sustained locomotion).
        dist_final = float(np.linalg.norm(pos_final[:2] - TARGET_POSITION[:2]))
        
        # Fitness (minimize): prioritize sustained forward gait over final position.
        # Heavily penalize robots that roll/fall initially then stop.
        fitness = dist_final - 1.0 * total_progress - 1.5 * max(0.0, sustained_progress) + 0.3 * lateral_drift
        return fitness

    def evolve(self) -> Individual | None:
        console.log("Initializing population...")
        population = [self.create_individual() for _ in range(POP_SIZE)]
        population = self.evaluate(population)

        ops = [
            EAStep("parent_selection", self.parent_selection),
            EAStep("reproduction", self.reproduction),
            EAStep("evaluation", self.evaluate),
            EAStep("survivor_selection", self.survivor_selection),
        ]

        ea = EA(population, operations=ops, num_of_generations=BUDGET)
        ea.run()
        return ea.get_solution("best", only_alive=False)


def main() -> None:
    console.rule("[bold purple]Starting Fast Joint Evolution (Tree Morph + Ctrl)[/bold purple]")
    evo = Evolution()
    best = evo.evolve()
    if best:
        console.rule("[bold green]Final Best Result[/bold green]")
        console.log(f"Best Fitness: {best.fitness:.4f}")
        if args.final_viewer != "none":
            evo.run_simulation(args.final_viewer, best, duration=DURATION)


if __name__ == "__main__":
    main()
