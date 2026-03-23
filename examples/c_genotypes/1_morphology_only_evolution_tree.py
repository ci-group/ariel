"""
Morphology-only evolution using TreeGenome and morphological descriptors.
Evolves robot structures based purely on morphological properties without simulation.
"""

# Standard library
import argparse

# --- ARIEL IMPORTS ---
import contextlib
import copy

# Standard library
import random
import time
from pathlib import Path

import mujoco

# Third-party libraries
# Third-party libraries
import numpy as np

# Function to show fitness landscape
from mujoco import viewer

# Pretty little errors and progress bars
from rich.console import Console
from rich.progress import track
from rich.traceback import install

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

# Local libraries
from ariel.ec import (
    EA,
    EAOperation,
    EASettings,
    Individual,
    Population,
)
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)

# tree genotype imports
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
from ariel.utils.morphological_descriptor import MorphologicalMeasures

# Initialize rich console
install()
console = Console()

# ============================================================================ #
#                               CONFIGURATION                                  #
# ============================================================================ #

parser = argparse.ArgumentParser(
    description="Morphology-Only Evolution (trees)",
)
parser.add_argument(
    "--budget",
    type=int,
    default=50,
    help="Number of generations",
)
parser.add_argument("--pop", type=int, default=100, help="Population size")
parser.add_argument(
    "--max-modules",
    type=int,
    default=20,
    help="Maximum modules per robot",
)
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Launch MuJoCo viewer for best individual",
)
parser.add_argument(
    "--max_modules",
    type=int,
    default=20,
    help="Maximum number of modules in a morphology",
)

args = parser.parse_args()

# Constants
POP_SIZE: int = args.pop
BUDGET: int = args.budget
NUM_MODULES: int = args.max_modules
MAX_DEPTH: int = 12  # Maximum tree depth to prevent bloat

# Type Aliases
# Population = list[Individual]

# Determinism
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

# Default spawn position for visualization
SPAWN_POSITION = (-0.8, 0.0, 0.1)


# ============================================================================ #
#                            MORPHOLOGICAL FITNESS                             #
# ============================================================================ #


def is_connected_tree(genome: TreeGenome) -> bool:
    """Check if genome is a valid connected tree with single root."""
    if len(genome.nodes) == 0:
        return False
    robot_graph = genome.to_networkx()
    # Check for single root (one node with no predecessors)
    roots = [n for n in robot_graph.nodes() if robot_graph.in_degree(n) == 0]
    if len(roots) != 1:
        return False
    # Check connectivity: all nodes reachable from root
    root = roots[0]
    reachable = set()
    stack = [root]
    while stack:
        node = stack.pop()
        reachable.add(node)
        stack.extend(
            succ
            for succ in robot_graph.successors(node)
            if succ not in reachable
        )
    return len(reachable) == robot_graph.number_of_nodes()


def calculate_morphological_fitness(genome: TreeGenome) -> float:
    """
    Calculate fitness based on morphological properties.

    Combines multiple morphological descriptors:
    - Symmetry (S): Higher symmetry is rewarded
    - Joints (J): More articulation is rewarded
    - Branching (B): Branching structure is rewarded
    - Limbs (L): Long limbs are rewarded

    Returns a fitness value (lower is better for minimization).
    """
    try:
        # Validate connectivity
        if not is_connected_tree(genome):
            return float("inf")

        robot_graph = genome.to_networkx()
        if robot_graph.number_of_nodes() == 0:
            return float("inf")

        measures = MorphologicalMeasures(robot_graph)

        # Weighted combination of morphological properties
        # Higher values are better, so we negate for minimization
        morpho_score = (
            measures.symmetry * 0.20  # Reward symmetry
            + measures.joints * 0.20  # Reward articulation
            + measures.branching * 0.20  # Reward branching
            + measures.length_of_limbs * 0.20  # Reward limb length
            + measures.module_diversity * 0.20  # Reward alternation/diversity
        )

        # Fitness = negative of morpho_score (for minimization framework)
        return -morpho_score

    except Exception:
        return float("inf")


def visualize_genome(genome: TreeGenome) -> None:
    """Construct a MuJoCo model from the genome and launch the viewer."""
    try:
        robot_graph = genome.to_networkx()
        if robot_graph.number_of_nodes() == 0:
            console.log("[red]Cannot visualize empty genome[/red]")
            return
        spec = construct_mjspec_from_graph(robot_graph).spec
        world = SimpleFlatWorldWithTarget()
        world.spawn(spec, position=SPAWN_POSITION)
        model = world.spec.compile()

        data = mujoco.MjData(model)
        viewer.launch(model=model, data=data)
    except Exception as e:
        console.log(f"[red]Visualization failed: {e}[/red]")


# ============================================================================ #
#                            EVOLUTION CLASS                                   #
# ============================================================================ #


class MorphologyEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,  # Minimize negative fitness (maximize morphological score)
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name="database.db",
        )

    # ------------------------------------------------------------------------ #
    #                          HELPER METHODS                                  #
    # ------------------------------------------------------------------------ #

    def get_joint_count(self, genome: TreeGenome) -> int:
        """Count the number of joints in a morphology."""
        try:
            robot_graph = genome.to_networkx()
            if robot_graph.number_of_nodes() == 0:
                return 0
            measures = MorphologicalMeasures(robot_graph)
            return measures.num_active_hinges
        except Exception:
            return 0

    def get_module_count(self, genome: TreeGenome) -> int:
        """Count the number of modules in a morphology."""
        return len(genome.nodes)

    def mutate_morphology(self, genome: TreeGenome) -> TreeGenome:
        """Apply mutations to morphology with multiple GP operators."""
        new = copy.deepcopy(genome)

        # Choose mutation type (standard GP mutation operators)
        mutation_type = RNG.choice(
            ["point", "subtree", "shrink", "hoist"], p=[0.4, 0.4, 0.1, 0.1],
        )

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
        if RNG.random() < 0.2:
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

    def crossover_morphologies(
        self, parent1: Individual, parent2: Individual,
    ) -> TreeGenome:
        """One-point crossover for morphologies. Recovers from invalid results."""
        t1 = parent1.genotype
        t2 = parent2.genotype
        if isinstance(t1, dict):
            t1 = TreeGenome.from_dict(t1)
        if isinstance(t2, dict):
            t2 = TreeGenome.from_dict(t2)

        child1, child2 = crossover_subtree(t1, t2)

        # Return first valid child, otherwise fallback to asexual copy
        chosen = child1 if RNG.random() < 0.5 else child2

        # If disconnected, return copy of a parent instead
        if not is_connected_tree(chosen):
            return TreeGenome.from_dict(random.choice([t1, t2]).to_dict())

        return chosen

    def create_individual(self) -> Individual:
        """Create a new individual with random morphology."""
        while True:
            # create random tree until it has modules
            genome = random_tree(NUM_MODULES)
            if self.get_module_count(genome) > 0:
                break
        ind = Individual()
        ind.genotype = genome.to_dict()
        ind.tags["ps"] = False
        ind.tags["valid"] = True
        return ind

    def reproduction(self, population: Population) -> Population:
        """Create offspring through crossover and mutation."""
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            console.log(
                "[yellow]Warning: No ps-tagged individuals, using entire population[/yellow]",
            )
            parents = population

        new_offspring: list[Individual] = []
        target_pool = self.config.target_population_size * 2
        while len(population) + len(new_offspring) < target_pool:
            use_sexual = len(parents) >= 2 and RNG.random() < 0.5
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                c_morph = self.crossover_morphologies(p1, p2)
            else:
                parent = random.choice(parents)
                c_morph = TreeGenome.from_dict(parent.genotype)

            # mutate morphology
            c_morph = self.mutate_morphology(c_morph)

            # body validation loop: ensure valid morphology
            valid_child = False
            attempts = 0
            while not valid_child and attempts < 20:
                has_modules = self.get_module_count(c_morph) > 0
                valid_depth = validate_tree_depth(c_morph, MAX_DEPTH)
                if has_modules and valid_depth:
                    valid_child = True
                else:
                    c_morph = self.mutate_morphology(c_morph)
                attempts += 1

            ind = Individual()
            ind.genotype = c_morph.to_dict()
            ind.tags["ps"] = False
            ind.tags["valid"] = True
            new_offspring.append(ind)

        population.extend(new_offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        """Evaluate population using morphological fitness."""
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        if not to_eval:
            return population

        for ind in track(to_eval, description="Evaluating..."):
            genome = TreeGenome.from_dict(ind.genotype)
            fitness = calculate_morphological_fitness(genome)
            ind.fitness = fitness
            ind.requires_eval = False

        return population

    def parent_selection(self, population: Population) -> Population:
        """Select top 50% as parents."""
        population = population.sort(sort="min")
        cutoff = len(population) // 2
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff
        ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
        console.log(
            f"[cyan]Parent Selection: {ps_count}/{len(population)} marked for reproduction[/cyan]",
        )
        return population

    def survivor_selection(self, population: Population) -> Population:
        """Keep top 50% as survivors."""
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        # Print statistics
        avg_fitness = np.mean([
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and ind.fitness_ != float("inf")
        ])
        min_fitness = min(
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and ind.fitness_ != float("inf")
        )
        max_fitness = max(
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and ind.fitness_ != float("inf")
        )

        console.log(
            f"[green]Survivor Selection:[/green] "
            f"Avg={avg_fitness:.4f}, Min={min_fitness:.4f}, Max={max_fitness:.4f}",
        )
        return population

    def evolve(self) -> Individual | None:
        """Run the evolutionary algorithm."""
        console.log("Initializing population...")
        population = Population([
            self.create_individual() for _ in range(POP_SIZE)
        ])

        # initial eval
        population = self.evaluate(population)

        ops = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]

        ea = EA(population, operations=ops, num_steps=BUDGET)
        ea.run()

        return ea.get_solution("best", only_alive=False)


def main() -> None:
    console.rule(
        "[bold purple]Starting Morphology-Only Evolution (Tree Genomes)[/bold purple]",
    )
    console.log(
        f"Population: {POP_SIZE}, Budget: {BUDGET}, Max Modules: {NUM_MODULES}",
    )

    evo = MorphologyEvolution()
    start_time = time.time()
    best = evo.evolve()
    elapsed = time.time() - start_time

    if best:
        console.rule("[bold green]Final Best Result[/bold green]")
        genome = TreeGenome.from_dict(best.genotype)
        fitness = calculate_morphological_fitness(genome)

        try:
            measures = MorphologicalMeasures(genome.to_networkx())
            console.log(f"Best Fitness Score: {fitness:.4f}")
            console.log(f"Modules: {measures.num_modules}")
            console.log(f"Joints: {measures.num_active_hinges}")
            console.log(f"Symmetry: {measures.symmetry:.4f}")
            console.log(f"Branching: {measures.branching:.4f}")
            console.log(f"Length of Limbs: {measures.length_of_limbs:.4f}")
            console.log(f"\nElapsed time: {elapsed:.2f}s")
        except Exception as e:
            console.log(f"[red]Error analyzing best individual: {e}[/red]")
    else:
        console.log("[red]No solution found![/red]")
    # Optionally visualize the best individual
    if best and args.visualize:
        try:
            visualize_genome(TreeGenome.from_dict(best.genotype))
        except Exception as e:
            console.log(f"[red]Visualization error: {e}[/red]")


if __name__ == "__main__":
    main()
