"""
Tree morphology version of 2_body_brain_evolution.
Uses TreeGenome for bodies with one-point crossover and node-replacement mutation.
"""

from __future__ import annotations

# Standard library
import argparse
import contextlib
import copy
import random
from pathlib import Path
from typing import Literal

import mujoco

# Third-party libraries
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# --- ARIEL IMPORTS ---
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)

# tree genotype imports
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    create_fully_connected_adjacency,
)
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

# Initialize rich console
install()
console = Console()

# ============================================================================ #
#                               CONFIGURATION                                  #
# ============================================================================ #

parser = argparse.ArgumentParser(
    description="Dual Evolution: Body + Brain (trees)",
)
parser.add_argument(
    "--budget", type=int, default=10, help="Number of generations",
)
parser.add_argument("--pop", type=int, default=30, help="Population size")
parser.add_argument("--dur", type=int, default=30, help="Sim Duration")
args = parser.parse_args()

# Constants
DURATION: int = args.dur
POP_SIZE: int = args.pop
BUDGET: int = args.budget
NUM_MODULES: int = 10
MAX_DEPTH: int = 12  # Increased from 8 to allow more complex morphologies
CTRL_GENOME_SIZE: int = NUM_MODULES * 5

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.5])

# Type Aliases
ViewerTypes = Literal["launcher", "video", "simple"]

# Determinism
SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


# ============================================================================ #
#                            EVOLUTION CLASS                                   #
# ============================================================================ #


class Evolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,  # Minimize Distance to Target
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            db_file_path=DATA / "database.db",
        )

    # ------------------------------------------------------------------------ #
    #                          HELPER METHODS                                  #
    # ------------------------------------------------------------------------ #
    def map_genotype_to_body(
        self, genome_data: dict | TreeGenome,
    ) -> mujoco.MjSpec | None:
        """Decodes TreeGenome into a MuJoCo Body Spec."""
        genome = (
            TreeGenome.from_dict(genome_data)
            if isinstance(genome_data, dict)
            else genome_data
        )

        try:
            robot_graph = genome.to_networkx()
            if robot_graph.number_of_nodes() == 0:
                return None
            return construct_mjspec_from_graph(robot_graph).spec
        except Exception:
            return None

    def map_genotype_to_brain(
        self, cpg: SimpleCPG, full_genome: list[float],
    ) -> None:
        # identical to original
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
            cpg.w.data.copy_(
                torch.from_numpy(0.2 + (3.8 * (p_w + 1.0) / 2.0)).float(),
            )
            cpg.amplitudes.data.copy_(
                torch.from_numpy(0.5 + (3.5 * (p_amp + 1.0) / 2.0)).float(),
            )
            cpg.ha.data.copy_(torch.from_numpy(p_ha * 2.0).float())
            cpg.b.data.copy_(torch.from_numpy(p_b * 0.5).float())

    def get_joint_count(self, genome: TreeGenome) -> int:
        spec = self.map_genotype_to_body(genome)
        if spec is None:
            return 0
        try:
            return spec.compile().nu
        except:
            return 0

    def mutate_ctrl_vector(self, genome: list[float]) -> list[float]:
        arr = np.array(genome)
        mask = RNG.random(arr.shape) < 0.35  # Increased to 35% mutation rate
        noise = RNG.normal(0, 0.5, arr.shape)  # Increased noise to 0.5
        arr[mask] += noise[mask]
        return np.clip(arr, -1.0, 1.0).tolist()

    def crossover_morphologies(
        self, parent1: Individual, parent2: Individual,
    ) -> TreeGenome:
        """One-point crossover that returns a single child genome."""
        t1 = parent1.genotype["morph"]
        t2 = parent2.genotype["morph"]
        if isinstance(t1, dict):
            t1 = TreeGenome.from_dict(t1)
        if isinstance(t2, dict):
            t2 = TreeGenome.from_dict(t2)
        child1, child2 = crossover_subtree(t1, t2)
        # pick one offspring at random
        return child1 if RNG.random() < 0.5 else child2

    def mutate_morphology(self, genome: TreeGenome) -> TreeGenome:
        new = copy.deepcopy(genome)

        # Choose mutation type (standard GP mutation operators)
        mutation_type = RNG.choice(
            ["point", "subtree", "shrink", "hoist"], p=[0.7, 0.2, 0.05, 0.05],
        )  # Favor point mutations heavily for stability

        if mutation_type == "point":
            # Point mutation: change node type/rotation
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            # Subtree mutation: replace subtree with new random tree
            mutate_subtree_replacement(new, max_modules=NUM_MODULES)
        elif mutation_type == "shrink":
            # Shrink mutation: replace node+subtree with single leaf
            mutate_shrink(new)
        elif mutation_type == "hoist":
            # Hoist mutation: promote child to replace parent
            mutate_hoist(new)

        # Additional rotation mutation (20% chance)
        if RNG.random() < 0.4:
            noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
            if noncore:
                nid = random.choice(noncore)
                # choose a new rotation allowed for its type
                mtype = ModuleType[new.nodes[nid]["type"]]
                rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
                if rots:
                    new.nodes[nid]["rotation"] = random.choice(rots)

        # prune any invalid edges before returning
        _prune_invalid_edges(new)
        with contextlib.suppress(ValueError):
            validate_genome_dict(new.to_dict())
        return new

    def crossover_ctrl_vectors(
        self, ctrl1: list[float], ctrl2: list[float],
    ) -> list[float]:
        """Uniform crossover for controller float vectors."""
        arr1 = np.array(ctrl1)
        arr2 = np.array(ctrl2)
        # Align sizes in case of mismatch
        size = max(len(arr1), len(arr2))
        if len(arr1) < size:
            arr1 = np.resize(arr1, size)
        if len(arr2) < size:
            arr2 = np.resize(arr2, size)
        # Uniform crossover: take each gene from either parent with 50% probability
        mask = RNG.random(size) < 0.5
        child = np.where(mask, arr1, arr2)
        return child.tolist()

    # rest of Evolution class same as original, replicating methods
    def create_individual(self) -> Individual:
        while True:
            # create random tree until it has joints
            genome = random_tree(NUM_MODULES)
            if self.get_joint_count(genome) > 0:
                break
        ind = Individual()
        ind.genotype = {
            "morph": genome.to_dict(),
            "ctrl": RNG.uniform(-1.0, 1.0, size=CTRL_GENOME_SIZE).tolist(),
        }
        ind.tags["ps"] = False
        ind.tags["valid"] = True
        ind.tags["debug_joints"] = 0
        return ind

    @EAOperation
    @staticmethod
    def reproduction(population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            console.log(
                "[yellow]Warning: No ps-tagged individuals, using entire population as parents[/yellow]",
            )
            parents = population
        new_offspring: list[Individual] = []
        target_pool = self.config.target_population_size * 2
        while len(population) + len(new_offspring) < target_pool:
            use_sexual = len(parents) >= 2 and RNG.random() < 0.5
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                c_morph = self.crossover_morphologies(p1, p2)
                c_ctrl = self.crossover_ctrl_vectors(
                    p1.genotype["ctrl"], p2.genotype["ctrl"],
                )
            else:
                parent = random.choice(parents)
                c_morph = TreeGenome.from_dict(parent.genotype["morph"])
                c_ctrl = (
                    parent.genotype["ctrl"].copy()
                    if isinstance(parent.genotype["ctrl"], list)
                    else parent.genotype["ctrl"]
                )
            # mutate morphology
            c_morph = self.mutate_morphology(c_morph)
            # Relaxed validation: only ensure basic validity, don't over-filter
            joint_count = self.get_joint_count(c_morph)
            if joint_count == 0:
                # If no joints, try one more mutation
                c_morph = self.mutate_morphology(c_morph)
                joint_count = self.get_joint_count(c_morph)
                if joint_count == 0:
                    # Still no joints, create new random tree
                    c_morph = random_tree(NUM_MODULES)

            c_ctrl = self.mutate_ctrl_vector(c_ctrl)

            ind = Individual()
            ind.genotype = {"morph": c_morph.to_dict(), "ctrl": c_ctrl}
            ind.tags["ps"] = False
            ind.tags["valid"] = True
            ind.tags["debug_joints"] = joint_count
            new_offspring.append(ind)
        population.extend(new_offspring)
        return population

    @EAOperation
    @staticmethod
    def evaluate(population: Population) -> Population:
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        if not to_eval:
            return population
        for ind in track(to_eval, description="Evaluating..."):
            fitness = self.run_simulation("simple", ind)
            ind.fitness = fitness
            ind.requires_eval = False
        return population

    @EAOperation
    @staticmethod
    def parent_selection(population: Population) -> Population:
        population.sort(
            key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"),
        )
        cutoff = len(population) // 2
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff
        ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
        console.log(
            f"[cyan]Parent Selection: {ps_count}/{len(population)} marked for reproduction[/cyan]",
        )
        return population

    @EAOperation
    @staticmethod
    def survivor_selection(population: Population) -> Population:
        population.sort(
            key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"),
        )
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False
        avg_fitness = np.mean([
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and ind.fitness_ != float("inf")
        ])
        console.log(
            f"[green]Survivor Selection: Avg fitness = {avg_fitness:.4f}[/green]",
        )
        return population

    def sync_ids(self, population: Population) -> None:
        # not needed for trees but stub to keep API
        pass

    def fast_physics_runner(
        self, model: mujoco.MjModel, data: mujoco.MjData, duration: float,
    ) -> None:
        steps_required = int(duration / model.opt.timestep)
        step = 0
        while step < steps_required:
            mujoco.mj_step(model, data)
            step += 1

    def run_simulation(self, mode: ViewerTypes, ind: Individual) -> float:
        mujoco.set_mjcb_control(None)
        expected_joints = ind.tags.get("debug_joints", 0)
        spec = None
        model = None
        attempts = 15 if mode != "simple" else 1
        for _ in range(attempts):
            temp_spec = self.map_genotype_to_body(ind.genotype["morph"])
            if temp_spec:
                try:
                    temp_model = temp_spec.compile()
                    if mode == "simple" or temp_model.nu == expected_joints:
                        spec = temp_spec
                        model = temp_model
                        break
                except:
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
            controller_callback_function=lambda m, d, *a, **k: cpg.forward(
                d.time,
            ),
            tracker=tracker,
        )
        ctrl.tracker.setup(world.spec, data)
        mujoco.set_mjcb_control(
            lambda m, d: ctrl.set_control(m, d, duration=DURATION),
        )
        mujoco.mj_resetData(model, data)
        match mode:
            case "simple":
                self.fast_physics_runner(model, data, duration=DURATION)
            case "video":
                recorder = VideoRecorder(
                    output_folder=str(DATA / "videos"),
                    file_name=f"dual_{ind.id}",
                )
                video_renderer(
                    model, data, duration=DURATION, video_recorder=recorder,
                )
            case "launcher":
                viewer.launch(model=model, data=data)
        delay_time = min(1.0, DURATION)
        delay_fraction = delay_time / DURATION if DURATION > 0 else 0
        dist = float("inf")
        if tracker.history["xpos"]:
            first_key = next(iter(tracker.history["xpos"].keys()))
            traj = tracker.history["xpos"][first_key]
            if traj:
                start_idx = max(0, int(len(traj) * delay_fraction))
                pos_after_delay = np.array(traj[start_idx])
                pos_final = np.array(traj[-1])
                valid_movement_vector = pos_final - pos_after_delay
                effective_pos = np.array(SPAWN_POSITION) + valid_movement_vector
                dist = np.sqrt(
                    np.sum((effective_pos[:2] - TARGET_POSITION[:2]) ** 2),
                )
        return dist

    def evolve(self) -> Individual | None:
        console.log("Initializing population...")
        population = Population([
            self.create_individual() for _ in range(POP_SIZE)
        ])
        # initial eval
        population = self.evaluate(population)
        ops = [
            self.parent_selection(),
            self.reproduction(),
            self.evaluate(),
            self.survivor_selection(),
        ]
        ea = EA(population, operations=ops, num_steps=BUDGET)
        ea.run()
        return ea.get_solution("best", only_alive=False)


def main() -> None:
    console.rule(
        "[bold purple]Starting Joint Evolution (Tree Morph + Ctrl)[/bold purple]",
    )
    evo = Evolution()
    best = evo.evolve()
    if best:
        console.rule("[bold green]Final Best Result")
        console.log(f"Best Fitness (Dist to Target): {best.fitness:.4f}")
        evo.run_simulation("launcher", best)


if __name__ == "__main__":
    main()
