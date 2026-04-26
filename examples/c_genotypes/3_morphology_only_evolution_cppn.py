"""
Morphology-only evolution for CPPN genomes using the same morphological fitness
as the tree-based example. Decodes CPPNs into module graphs using the
best-first decoder and evaluates with MorphologicalMeasures.
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Any, List

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# Initialize rich console
install()
console = Console()

from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.body_phenotypes.robogen_lite.decoders.score_cube import MorphologyDecoderCubePruning
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
import mujoco
from mujoco import viewer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Morphology-only CPPN evolution")
parser.add_argument("--budget", type=int, default=50)
parser.add_argument("--pop", type=int, default=100)
parser.add_argument("--max-modules", type=int, default=15)
parser.add_argument("--visualize", type=bool, default=True, help="Launch MuJoCo viewer for best individual")
args = parser.parse_args()

POP_SIZE = args.pop
BUDGET = args.budget
NUM_MODULES = args.max_modules

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

# Default spawn position for visualization
SPAWN_POSITION = (-0.8, 0.0, 0.1)

# CPPN input/output sizes (6 inputs: parent type, parent rotation, face; outputs: 1 for add/remove decision, T for module type, R for rotation)
T = NUM_OF_TYPES_OF_MODULES
R = NUM_OF_ROTATIONS
NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + T + R

# Id manager for mutations
id_manager = IdManager(node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
                       innov_start=(NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS) - 1)

# ---------------------------------------------------------------------------
# Fitness (same weighting as tree example)
# ---------------------------------------------------------------------------

def morpho_score_from_graph(graph) -> float:
    try:
        measures = MorphologicalMeasures(graph)
        score = (
            measures.symmetry * 0.20
            + measures.joints * 0.20
            + measures.branching * 0.20
            + measures.length_of_limbs * 0.20
            + measures.module_diversity * 0.20
        )
        return score
    except Exception:
        return float("nan")


def visualize_genome(cppn_genome: Genome) -> None:
    """Decode a CPPN genome to a graph, build an MJCF spec and launch MuJoCo viewer."""
    try:
        decoder = MorphologyDecoderBestFirst(cppn_genome=cppn_genome, max_modules=NUM_MODULES)
        # decoder = MorphologyDecoderCubePruning(cppn_genome=cppn_genome, max_modules=NUM_MODULES)
        graph = decoder.decode()
        if graph.number_of_nodes() == 0:
            console.log("[red]Cannot visualize empty decoded graph[/red]")
            return
        spec = construct_mjspec_from_graph(graph).spec
        world = SimpleFlatWorldWithTarget()
        world.spawn(spec, position=SPAWN_POSITION)
        model = world.spec.compile()
        data = mujoco.MjData(model)
        viewer.launch(model=model, data=data)
    except Exception as e:
        console.log(f"[red]Visualization failed: {e}[/red]")

# ---------------------------------------------------------------------------
# Evolution class for CPPN genomes
# ---------------------------------------------------------------------------
class CPPNEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_of_generations=BUDGET,
            target_population_size=POP_SIZE,
            db_file_path=DATA / "database.db",
        )

    def create_random_genome(self) -> Genome:
        # create a random base genome
        g = Genome.random(
            num_inputs=NUM_CPPN_INPUTS,
            num_outputs=NUM_CPPN_OUTPUTS,
            next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
            next_innov_id=0,
        )
        # apply a few structural mutations initially
        for _ in range(3):
            g.mutate(0.6, 0.6, id_manager.get_next_innov_id, id_manager.get_next_node_id)
        return g

    def create_individual(self) -> Individual:
        genome = self.create_random_genome()
        ind = Individual()
        ind.genotype = {"cppn": genome.to_dict()}
        ind.tags["ps"] = False
        ind.tags["valid"] = True
        return ind

    def decode_to_graph(self, genome: Genome):
        decoder = MorphologyDecoderBestFirst(cppn_genome=genome, max_modules=NUM_MODULES)
        return decoder.decode()

    def evaluate(self, population: List[Individual]) -> List[Individual]:
        to_eval = [ind for ind in population if ind.alive and ind.tags.get("valid") and ind.requires_eval]
        if not to_eval:
            return population
        for ind in track(to_eval, description="Evaluating..."):
            cppn = Genome.from_dict(ind.genotype["cppn"])
            graph = self.decode_to_graph(cppn)
            score = morpho_score_from_graph(graph)
            # EA expects minimization; store negative score
            ind.fitness = -score if not np.isnan(score) else float("inf")
            ind.requires_eval = False
        return population

    def mutate(self, genome: Genome) -> Genome:
        g = genome.copy()
        # structural mutations
        g.mutate(0.2, 0.3, id_manager.get_next_innov_id, id_manager.get_next_node_id)
        return g

    def crossover(self, a: Genome, b: Genome) -> Genome:
        child = a.crossover(b)
        return child

    def reproduction(self, population: List[Individual]) -> List[Individual]:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = population
        new_offspring: List[Individual] = []
        target_pool = self.config.target_population_size * 2
        while len(population) + len(new_offspring) < target_pool:
            if len(parents) >= 2 and RNG.random() < 0.5:
                p1, p2 = random.sample(parents, 2)
                g1 = Genome.from_dict(p1.genotype["cppn"]) if isinstance(p1.genotype["cppn"], dict) else p1.genotype["cppn"]
                g2 = Genome.from_dict(p2.genotype["cppn"]) if isinstance(p2.genotype["cppn"], dict) else p2.genotype["cppn"]
                child = self.crossover(g1, g2)
            else:
                parent = random.choice(parents)
                child = Genome.from_dict(parent.genotype["cppn"]) if isinstance(parent.genotype["cppn"], dict) else parent.genotype["cppn"]
            child = self.mutate(child)
            ind = Individual()
            ind.genotype = {"cppn": child.to_dict()}
            ind.tags["ps"] = False
            ind.tags["valid"] = True
            new_offspring.append(ind)
        population.extend(new_offspring)
        return population

    def parent_selection(self, population: List[Individual]) -> List[Individual]:
        population.sort(key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"))
        cutoff = len(population) // 2
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff
        console.log(f"[cyan]Parent Selection: {sum(1 for ind in population if ind.tags.get('ps', False))}/{len(population)} marked[/cyan]")
        return population

    def survivor_selection(self, population: List[Individual]) -> List[Individual]:
        population.sort(key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf"))
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False
        return population

    def evolve(self) -> Individual | None:
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


if __name__ == "__main__":
    console.log(f"Starting CPPN morphology-only evolution: pop={POP_SIZE}, gen={BUDGET}")
    evo = CPPNEvolution()
    start = time.time()
    best = evo.evolve()
    elapsed = time.time() - start
    if best:
        cppn = Genome.from_dict(best.genotype["cppn"])
        graph = evo.decode_to_graph(cppn)
        measures = MorphologicalMeasures(graph)
        score = morpho_score_from_graph(graph)
        console.log(f"Best morphological score: {score:.4f}")
        console.log(f"Modules: {measures.num_modules}, Joints: {measures.num_active_hinges}, Symmetry: {measures.symmetry:.4f}, Diversity: {measures.module_diversity:.4f}")
        console.log(f"Elapsed: {elapsed:.2f}s")
        # Optionally visualize the best individual
        if args.visualize:
            try:
                visualize_genome(cppn)
            except Exception as e:
                console.log(f"[red]Visualization error: {e}[/red]")
    else:
        console.log("No solution found")
